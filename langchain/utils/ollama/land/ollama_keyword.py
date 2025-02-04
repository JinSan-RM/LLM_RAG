from config.config import OLLAMA_API_URL
import requests
import json
import re


class OllamaKeywordClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.2, model: str = ''):
        self.api_url = api_url
        self.temperature = temperature
        self.model = model

    async def send_request(self, prompt: str) -> str:
        """
        공통 요청 처리 함수: /generate API 호출 및 응답처리
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "format": "json"
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
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

    async def process_menu_data(self, menu_data: str) -> list:
        """
        LLM의 응답에서 JSON 형식만 추출 및 정리 (리스트 형식으로)
        """
        try:
            # JSON 부분만 추출 (정규표현식 사용) - 리스트 형식으로 수정
            json_match = re.search(r"\[.*\]", menu_data, re.DOTALL)
            if not json_match:
                raise ValueError("JSON 형식을 찾을 수 없습니다.")

            # JSON 텍스트 추출
            json_text = json_match.group()

            # 이중 따옴표 문제 수정 (선택적)
            json_text = json_text.replace("'", '"').replace("\n", "").strip()

            # JSON 파싱
            json_data = json.loads(json_text)

            # 리스트 형태로 반환
            return json_data

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {e}")
            raise RuntimeError("menu_data의 형식이 올바르지 않습니다.") from e

    async def section_keyword_recommend(self, section: str, context: str):
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        You are a professional designer who searches for images that will fit in each section of the website landing page.
        Choose two **English keywords** to search for images, taking into account the full data and each section summary data for each section configuration of the configured landing page.

        ###Rules
        1. Please extract keywords that match the section.
        {section}
        2. The purpose of extracting these keywords is to find terms that perform well on image search sites.
        3. Please consider the data and specify one keyword that is favorable for image search.
            - Each keyword must consist of up to two words.
            - The selected keywords will be common to all menus.
            - Ensure there are no typos.
            - Write in English only.
            - Do not use any special characters, symbols, or underscores.
            - Each keyword must consist of letters and spaces only (no numbers, punctuation, or special symbols).
        4. Do not write any explanations, sentences, comments, or code blocks other than the keyword list.

        5. Output format:
        ['example', 'example2']

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Input Data:
        {context}

        <|start_header_id|>assistant<|end_header_id|>
        Always follow the data in the output format and return the results only in the form of **keyword list**
        Be sure to return all data in the output format only in **English**
        """
        keyword_data = await self.send_request(prompt=prompt)
        print("Let's see keyword_data : ", keyword_data)
        return keyword_data

    async def section_keyword_create_logic(self, section: str, context: str):
        """
        data, summary, section을 이용해서 keyword를 생성하는 로직.
        """
        try:
            section_context = await self.section_keyword_recommend(
                section,
                context
                )
            # JSON 데이터 파싱
            section_data_with_keyword = await self.process_menu_data(section_context)
            section_data_with_keyword = await self.process_data(section_data_with_keyword)
            return section_data_with_keyword

        except Exception as e:  # 모든 예외를 잡고 싶다면
            print(f"Error processing landing structure: {e}")
            return "error"
        
    # 특수문자 제거 함수 (공백 유지)
    def clean_keyword(self, keyword):
        cleaned = re.sub(r'[^\w\s]', '', keyword)  # 단어 문자와 공백만 허용 (언더바 포함)
        cleaned = re.sub(r'_', ' ', cleaned)  # 언더바를 공백으로 변환
        cleaned = re.sub(r'-', ' ', cleaned)  # 언더바를 공백으로 변환
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # 여러 개의 공백을 하나로 줄이고 앞뒤 공백 제거
        return cleaned
    
    # JSON 데이터 파싱 및 정리
    async def process_data(self, section_context):

        # 키워드 정리
        if isinstance(section_context, dict):  # 딕셔너리 형태라면
            section_context = {
                key: self.clean_keyword(value) if isinstance(value, str) else value
                for key, value in section_context.items()
            }
        elif isinstance(section_context, list):  # 리스트 형태라면
            section_context = [
                self.clean_keyword(item) if isinstance(item, str) else item
                for item in section_context
            ]

        print(section_context, "<====keyword")
        return section_context