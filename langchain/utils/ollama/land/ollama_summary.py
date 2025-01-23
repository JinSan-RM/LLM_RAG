from config.config import OLLAMA_API_URL
import requests
import json
from typing import List


class OllamaSummaryClient:

    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.25, model: str = ''):
        self.api_url = api_url
        self.temperature = temperature
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
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=15)
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
            raise RuntimeError(f"Ollama API 요청 실패: {e}") from e

    def split_into_chunks(self, data: str, max_length: int) -> List[str]:
        """
        데이터를 최대 길이에 맞게 청크로 분할하는 함수
        :param data: 대용량 입력 데이터 문자열
        :param max_length: 각 청크의 최대 문자 수
        :return: 분할된 청크 리스트
        """
        return [data[i:i + max_length] for i in range(0, len(data), max_length)]

    async def summarize_chunk(self, chunk: str, desired_summary_length: int, previous_summary: str = "") -> str:
        """
        단일 청크를 요약하는 비동기 함수
        :param chunk: 요약할 청크 문자열
        :param previous_summary: 이전에 요약된 내용 (옵션)
        :return: 요약된 문자열
        """

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
        # 최종 요약 길이에 맞게 자름
        if len(combined_summary) > final_summary_length:
            combined_summary = combined_summary[:final_summary_length]
        else:
            print("[backpropagation_summary] Combined summary final_summary_length")
        return combined_summary

    async def summary_proposal(self, final_summary: str):

        prompt = f"""
            <|start_header_id|>system<|end_header_id|>
            당신은 내용 요약을 도와주는 AI입니다.
            다음 **규칙**을 반드시 지켜서 내가 원하는 proposal 생성해줘

            1) "assistant"처럼 **생성**해야 하고, 규정된 형식을 **절대** 벗어나면 안 된다.
            2) **웹사이트명, 키워드, 제작목적, 주요타겟층, 핵심가치, 주요서비스, 부가기능**을 **필수**키로 사용해라.

                - **키워드, 주요타겟층, 핵심가치, 주요성과**는 []형식을 사용해서 입력해라.
                - 입력데이터를 토대로 사이트명을 지정해라.
                - 입력데이터를 토대로 핵심 키워드 **7개**를 [] 형태로 생성해주세요.
                - 입력데이터를 토대로 제작목적을 **200자**로 생성해주세요.
                - 입력데이터를 토대로 사이트 제작자의 **주요타겟층 3개**를 설정해서 [] 형태로 생성해주세요.
                - 입력데이터를 토대로 기업의 **핵심가치를 생성**해주세요.
                - 입력데이터를 토대로 기업의 서비스 명 및 설명 추출 **string**으로 생성해주세요.
                - 입력데이터를 토대로 기업의 주요성과에 알맞은 수상내역 또는 성과 **5개 내외**로 생성해주세요.
            3) 모든 값들은 반드시 문자열이야한다.
            4) **출력은 오직 JSON 형태**로만 해야 하며, 그 외 어떤 설명(문장, 코드, 해설)도 삽입하지 말 것.
            5) 모든 텍스트 내용은 입력 데이터에 맞춰 작성하고 목적/흐름을 고려해 자연스럽게 작성한다.
            6) 아래 예시 구조를 준수해라.
            7) **출력 형식 예시** (JSON 구조 예시):

                {{
                    "웹사이트명": "example 랜딩페이지",
                    "키워드": ["","","","","","",""],
                    "제작목적": "100자 이내",
                    "주요타겟층": ["","",""],
                    "핵심가치": ["", "", ""],
                    "주요서비스": "",
                    "주요성과":["","","","",""]
                }}

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            입력 데이터:
            {final_summary}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            -입력데이터를 기준으로 출력을 **Json** 형태로만 반환하세요.
        """
        return await self.send_request(prompt=prompt)

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

        current_chunk_number = 0
        remaining_data = data
        summarized_chunks = []
        accumulated_summary_length = 0
        previous_summary = ""

        while remaining_data and accumulated_summary_length < final_summary_length:
            # 현재 청크 크기 계산
            current_chunk_size = max_tokens_per_chunk - (w * current_chunk_number)
            current_chunk_size = max(current_chunk_size, 100)

            # 청크 분할
            chunks = self.split_into_chunks(remaining_data, current_chunk_size)

            if not chunks:
                print("[store_chunks] No more chunks to process.")
                break

            # 첫 번째 청크 가져오기
            current_chunk = chunks[0]
            remaining_data = ''.join(chunks[1:])  # 나머지 데이터

            # 마지막 요약 길이를 계산
            remaining_summary_space = final_summary_length - accumulated_summary_length
            if remaining_summary_space <= 0:
                print("[store_chunks] Remaining summary space is 0. Stopping.")
                break
            # 현재 요약 목표 설정
            desired_summary_length = min(500 + (w * current_chunk_number), remaining_summary_space)
            max_tokens = min(max_tokens_per_chunk - (w * current_chunk_number), remaining_summary_space)
            max_tokens = max(max_tokens, 100)  # 최소 100은 보장

            # 청크 요약 (이전 요약을 포함)
            summary = await self.summarize_chunk(
                current_chunk,
                desired_summary_length,
                previous_summary
                )
            if not summary:
                print("[store_chunks] Summary failed current chunk.")
                continue

            if len(summary) > remaining_summary_space:
                summary = summary[:remaining_summary_space]

            summarized_chunks.append(summary)
            accumulated_summary_length += len(summary)

            previous_summary = summary  # 현재 요약을 다음 청크에 포함

            current_chunk_number += 1

        # 모든 요약된 청크를 합쳐 최종 요약 생성
        final_summary = ' '.join(summarized_chunks)  # 이미 초과 방지됨

        # proposal = await self.summary_proposal(final_summary)

        return final_summary
