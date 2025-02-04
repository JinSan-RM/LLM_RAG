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
        당신은 웹사이트 랜딩 페이지의 각 섹션에 들어갈 이미지를 검색하는 전문 디자이너입니다.
        구성된 랜딩 페이지의 각 섹션 구성에 알맞게 전체 데이터와 각 섹션 요약 데이터를를 고려하여 이미지를 검색할 영문 키워드를 2개 고르세요.

        ### 규칙
        1. **섹션**에 어울리는 키워드를 추출하여 주세요.
            {section}
        2. 아래의 키워드를 추출하는 이유는 이미지 검색 사이트에서 이미지가 잘 검색될 키워드를 찾는 것을 잊지마.
        3. 데이터를 고려하여 이미지 검색에 유리한 키워드를 1개 지정해줘.
         - 각 키워드는 최대 3개의 단어로 이루어져있어.
         - 여기서 선정된 키워드는 모든 메뉴에 공통적으로 들어갈거야.
         - 오탈자가 없게 작성해줘.
         - 영어로만 작성해줘.
         - 각 키워드는 최대 3개의 단어로 이루어져있어.
        4. **키워드 리스트** 이외에 어떤 설명, 문장, 주석, 코드 블록도 작성하지 마세요.

        5. 출력형식:
        ['example', 'example2', 'example3']

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {context}

        <|start_header_id|>assistant<|end_header_id|>
        반드시 출력 형식의 데이터를 따라 **키워드 리스트** 형태로만 결과를 반환
        반드시 출력 형식의 데이터는 모두 **영어**로만 반환
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
            print(section_data_with_keyword, "<====keyword")
            return section_data_with_keyword

        except Exception as e:  # 모든 예외를 잡고 싶다면
            print(f"Error processing landing structure: {e}")
            return "error"
