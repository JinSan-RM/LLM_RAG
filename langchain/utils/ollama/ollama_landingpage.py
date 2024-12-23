from utils.ollama.ollama_content import OllamaContentClient
from utils.ollama.ollama_landingpage_plan import WebsitePlan, WebsitePlanException
from config.config import OLLAMA_API_URL
import requests, json
from typing import List, Dict


content_client = OllamaContentClient()

class OllamaLandingClient:
    
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.4, n_ctx = 4096, max_token = 4096, model:str = ''):
        self.api_url = api_url
        self.temperature = temperature
        self.n_ctx = n_ctx
        self.max_token = max_token
        self.model = model
        
    async def send_request(self, prompt: str) -> str:
        """
        공통 요청 처리 함수 : API 호출 및 응답 처리
        
        Generate 버전전
        """
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "n_ctx": self.n_ctx,
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리

            full_response = response.text  # 전체 응답
            lines = full_response.splitlines()
            all_text = ""
            for line in lines:
                try:
                    json_line = json.loads(line.strip())  # 각 줄을 JSON 파싱
                    all_text += json_line.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    continue  # JSON 파싱 오류 시 건너뛰기
                
            return all_text.strip() if all_text else "Empty response received"

        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 실패: {e}")
            raise RuntimeError(f"Ollama API 요청 실패: {e}")
        
        
    def split_into_chunks(self, data: str, max_length: int) -> List[str]:
        """
        데이터를 최대 길이에 맞게 청크로 분할하는 함수
        :param data: 대용량 입력 데이터 문자열
        :param max_length: 각 청크의 최대 문자 수
        :return: 분할된 청크 리스트
        """
        return [data[i:i + max_length] for i in range(0, len(data), max_length)]
    
    async def summarize_chunk(self, chunk: str, max_tokens: int, desired_summary_length: int, previous_summary: str = "") -> str:
        """
        단일 청크를 요약하는 비동기 함수
        :param chunk: 요약할 청크 문자열
        :param max_tokens: 요약 시 사용할 최대 토큰 수
        :param previous_summary: 이전에 요약된 내용 (옵션)
        :return: 요약된 문자열
        """
        print(f"previous_summary : {len(previous_summary)}   {previous_summary}\n")
        print(f"chunk : {len(chunk)}   {chunk}\n")
        print(f"desired_summary_length : {desired_summary_length}")
        
        if previous_summary:
            combined_text = f"{previous_summary}\n\n{chunk}"
        else:
            combined_text = chunk
        
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        - 입력된 데이터를 {desired_summary_length}자로 요약해주세요.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {combined_text}
        <|start_header_id|>assistant<|end_header_id|>
        - 입력된 데이터를 {desired_summary_length}자로 요약해주세요.
        """
        print(f"[summarize_chunk] Prompt prepared for summary. Length: {len(prompt)} characters")
        try:
            # API 요청 (비동기 함수이므로 await 사용)
            summary = await self.send_request(prompt=prompt)
            return summary
        except Exception as e:
            print(f"청크 요약 중 오류 발생: {e}")
            return ""
    
    def backpropagation_summary(self, summaries: List[str], final_summary_length: int) -> str:
        """
        요약된 청크들을 최종 요약으로 합치는 함수
        :param summaries: 요약된 청크 리스트
        :param final_summary_length: 최종 요약의 최대 문자 수
        :return: 최종 요약 문자열
        """
        # 모든 요약을 합침
        combined_summary = ' '.join(summaries)
        print(f"[backpropagation_summary] Combined summary length before trimming: {len(combined_summary)}")
        
        # 최종 요약 길이에 맞게 자름
        if len(combined_summary) > final_summary_length:
            combined_summary = combined_summary[:final_summary_length]
            print(f"[backpropagation_summary] Trimmed combined summary to {final_summary_length} characters")
        else:
            print(f"[backpropagation_summary] Combined summary is within the final_summary_length")
        
        return combined_summary
    
    async def store_chunks(self, data: str, model_max_token: int, final_summary_length: int, max_tokens_per_chunk: int) -> str:
        """
        대용량 데이터를 청크로 분할하고, 각 청크를 모델에 전달하여 요약한 후, 모든 요약을 합쳐 최종 요약을 생성하는 함수

        :param model_max_token: 모델의 최대 토큰 수
        :param data: 대용량 입력 데이터 문자열
        :param final_summary_length: 최종 요약의 최대 문자 수
        :param max_tokens_per_chunk: 각 청크의 최대 토큰 수
        :return: 최종 요약 문자열
        """
        # w 값 설정
        if model_max_token == 8192:
            w = 250
        elif model_max_token == 4096:
            w = 100
        else:
            w = 100  # 기본값 설정
        print(f"[store_chunks] Model max token: {model_max_token}, w: {w}")
        
        current_chunk_number = 0
        remaining_data = data
        summarized_chunks = []
        current_summary_target = 500
        accumulated_summary_length = 0
        previous_summary = ""
        
        while remaining_data and current_summary_target <= final_summary_length:
            # 지금 현재의 청크 크기를 계산
            current_chunk_size = max_tokens_per_chunk - (w * current_chunk_number)
            current_chunk_size = max(current_chunk_size, 100)
            print(f"[store_chunks] Current chunk number: {current_chunk_number}, chunk size: {current_chunk_size}")
            
            # 청크 분할
            chunks = self.split_into_chunks(remaining_data, current_chunk_size)
            
            if not chunks:
                print("[store_chunks] No more chunks to process.")
                break
            
            # 첫 번째 청크 가져오기
            current_chunk = chunks[0]
            remaining_data = ''.join(chunks[1:])  # 나머지 데이터
            print(f"[store_chunks] Processing chunk {current_chunk_number + 1} with size {len(current_chunk)} characters")
            
            # 현재 요약 목표에 맞춰 max_tokens 설정
            desired_summary_length = current_summary_target
            max_tokens = max_tokens_per_chunk - (w * current_chunk_number)
            max_tokens = max(max_tokens, 100)
            print(f"[store_chunks] Desired summary length: {desired_summary_length}, max_tokens: {max_tokens}")
            
            # 청크 요약 (이전 요약을 포함)
            summary = await self.summarize_chunk(current_chunk, max_tokens, desired_summary_length, previous_summary)
            if not summary:
                print("[store_chunks] Summary failed for current chunk. Skipping to next.")
                continue
            
            summarized_chunks.append(summary)
            accumulated_summary_length += len(summary)
            print(f"[store_chunks] Accumulated summary length: {accumulated_summary_length}")
            previous_summary = summary  # 현재 요약을 다음 청크에 포함
            
            # 현재 요약 목표에 도달했는지 확인
            if accumulated_summary_length >= current_summary_target:
                print(f"[store_chunks] Reached current summary target: {current_summary_target} characters")
                # 다음 청크 번호 및 요약 목표 설정
                current_chunk_number += 1
                current_summary_target = 500 + (w * current_chunk_number)
                # 최종 요약 길이를 초과하지 않도록 조정
                if current_summary_target > final_summary_length:
                    current_summary_target = final_summary_length
                print(f"[store_chunks] Updated summary target to: {current_summary_target} characters")
        
        # 모든 요약된 청크를 합쳐 최종 요약 생성
        print("[store_chunks] Combining all summarized chunks into final summary.")
        final_summary = self.backpropagation_summary(summarized_chunks, final_summary_length)
        print(f"[store_chunks] Final summary length: {len(final_summary)} characters")
        return final_summary

        # 1. 데이터 청크로 분할
        chunks = self.split_into_chunks(data, max_tokens_per_chunk)
        summarized_chunks = []
        cycle_chunk = max_tokens / len(chunks)
        print("\n=============>", len(chunks), type(chunks), cycle_chunk,"<================cycle_chunk\n")
        # 2. 각 청크를 요약
        for idx, chunk in enumerate(chunks):
            # 메시지 형식으로 변환
            prompt = f"""
            <|start_header_id|>system<|end_header_id|>
            - 입력된 데이터를 요약해주세요.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            입력 데이터:
            {chunk}
            """
            try:
                # API 요청 (비동기 함수이므로 await 사용)
                response = await self.send_request(prompt=prompt)

                summarized_chunks.append(response)
                print(f"Chunk {idx + 1} 요약 완료. \n {response}")
            
            except Exception as e:
                print(f"Chunk {idx + 1} 처리 중 오류 발생: {e}")

        # 3. 모든 요약을 하나로 결합
        combined_summaries = ' '.join(summarized_chunks)
        
        # 4. 최종 요약 요청
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        - 입력된 내용을 기반으로 {max_tokens}자 이내로 기획서를 작성해 주세요.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {combined_summaries}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        - 입력된 내용을 기반으로 {max_tokens}자 이내로 기획서를 작성해 주세요.
        """
    
        try:
            # final_websiteplan = WebsitePlan( **combined_summaries )
            # summarized_chunks.append(website_plan.dict())
            final_response = await self.send_request(prompt=prompt)
            print(final_response,"<====final_response")

            # 최종 요약이 500자를 초과하지 않도록 확인
            if len(final_response) > final_summary_length:
                final_response = final_response[:final_summary_length]
                print("최종 요약이 500자를 초과하여 잘랐습니다.")

            print("최종 요약 완료.")
            return final_response

        except Exception as e:
            print(f"최종 요약 처리 중 오류 발생: {e}")
            return ""
        
    async def generate_section(self, model: str,summary:str, section_name: str) -> str:
        """
        랜딩 페이지 섹션을 생성하는 함수
        """
        prompt = f"""
            <|start_header_id|>system<|end_header_id|>
            - 너는 사이트의 섹션 구조를 정해주고, 그 안에 들어갈 내용을 작성해주는 AI 도우미야.
            - 입력된 데이터를 기준으로 단일 페이지를 갖는 랜딩사이트 콘텐츠를 생성해야 해.
            - 'children'의 컨텐츠 내용의 수는 너가 생각하기에 섹션에 알맞게 개수를 수정해서 생성해줘.
            - 섹션 '{section_name}'에 어울리는 내용을 생성해야 하며, 반드시 다음 규칙을 따라야 한다:
            1. assistant처럼 생성해야 하고 형식을 **절대** 벗어나면 안 된다.
            2. "div, h1, h2, h3, p, ul, li" 태그만 사용해서 섹션의 콘텐츠를 구성해라.
            3. 태그들 안의 css나 다른것들은 입력하지마세요.
            4. 섹션 안의 `children` 안의 컨텐츠 개수는 2~10개 사이에서 자유롭게 선택하되, 내용이 반복되지 않도록 다양하게 생성하라.
            5. 모든 텍스트 내용은 입력 데이터에 맞게 작성하고, 섹션의 목적과 흐름에 맞춰야 한다.
            6. 출력 결과는 코드 형태만 허용된다. 코드는 **절대 생성하지 마라.**
            7. 오직 한글로만 작성하라.
        

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            입력 데이터:
            {summary}
            랜딩 페이지 섹션을 구성하기 위한 PDF 내용 전체가 에 포함되어 있습니다.
            섹션:
            {section_name}


            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            - <div class = {section_name}>으로 시작해야하며, 이 안의 내용을 채워야한다. </div>
            - 너는 출력 예시와 같은 코드 구조 응답만을 반환해야 한다.
            - 출력 예시 :
            <div class="{section_name}_SECTION">                                                                                                                                     
                <h1>회사 이름 혹은 제품 이름</h1>
                <p>회사 소개 혹은 제품 소개</p>
            </div>
            - 출력 예시 :                                    
            <div class="{section_name}_SECTION">
                <h3>서비스 강점</h3>
                <h1>서비스 주요 특징</h1>
                <p>서비스 주요 특징 설명</p>
                <ul>                                                                                                                                 
                    <li>
                        <h2>특징 1</h2>
                        <p>특징 1에 대한 설명</p>
                    </li>
                    <li>
                        <h2>특징 2</h2>
                        <p>특징 2에 대한 설명</p>
                    </li>
                    <li>
                        <h2>특징 3</h2>
                        <p>특징 3에 대한 설명</p>
                    </li>
                </ul>
            </div>                            
        """
        return await self.send_request(prompt=prompt)  