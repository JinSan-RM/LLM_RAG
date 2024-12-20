from utils.ollama.ollama_content import OllamaContentClient
from config.config import OLLAMA_API_URL
import requests, json


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
        
        
    def split_into_chunks(self, data: str, max_tokens: int) -> list:
        """
        데이터를 최대 토큰 수에 맞게 청크로 분할하는 함수

        :param data: 입력 데이터 문자열
        :param max_tokens: 각 청크의 최대 토큰 수
        :return: 분할된 청크 리스트
        """
        paragraphs = data.split('\n\n')
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk.split()) + len(para.split()) <= max_tokens:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    
    async def temp_store_chunks(self, model: str, data: str) -> str:
        """
        대용량 데이터를 청크로 분할하고, 각 청크를 모델에 전달하여 요약한 후, 모든 요약을 합쳐 최종 500자 요약을 생성하는 함수

        :param model: 사용하려는 모델 이름
        :param data: 대용량 입력 데이터 문자열
        :return: 최종 500자 요약 문자열
        """
        content_client = OllamaContentClient()
        max_tokens_per_chunk = 1000  # 각 청크의 최대 토큰 수 (예시)
        final_summary_length = 500    # 최종 요약의 최대 문자 수

        # 1. 데이터 청크로 분할
        chunks = self.split_into_chunks(data, max_tokens_per_chunk)
        summarized_chunks = []

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
                response = await content_client.send_request(prompt)
                

                summarized_chunks.append(response)
                print(f"Chunk {idx + 1} 요약 완료.")
            
            except Exception as e:
                print(f"Chunk {idx + 1} 처리 중 오류 발생: {e}")

        # 3. 모든 요약을 하나로 결합
        combined_summaries = ' '.join(summarized_chunks)
        
        # 4. 최종 요약 요청
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        - 모든 요약된 내용을 기반으로 500자 이내로 다시 요약해 주세요.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {combined_summaries}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        - 500자 이내로 요약해 주세요.
        """
    
        try:
            final_response = await content_client.send_request(prompt)
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
        return await self.send_request(prompt)  