from config.config import OLLAMA_API_URL
import requests, json, re

class OllamaKeywordClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.1, model: str =''):
        self.api_url = api_url
        self.temperature = temperature
        self.model = model
        
    async def send_request(self, prompt: str) -> str:
        """
        공통 요청 처리 함수: /generate API 호출 및 응답처리
        """
        payload = {
            "model" : self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "format": "json"
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

    async def process_menu_data(self, menu_data: str) -> dict:
        """
        LLM의 응답에서 JSON 형식만 추출 및 정리
        """
        try:
            # JSON 부분만 추출 (정규표현식 사용)
            json_match = re.search(r"\{.*\}", menu_data, re.DOTALL)
            if not json_match:
                raise ValueError("JSON 형식을 찾을 수 없습니다.")
            
            # JSON 텍스트 추출
            json_text = json_match.group()

            # 이중 따옴표 문제 수정 (선택적)
            json_text = json_text.replace("'", '"').replace("\n", "").strip()

            # JSON 파싱
            json_data = json.loads(json_text)
            return json_data

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {e}")
            raise RuntimeError("menu_data의 형식이 올바르지 않습니다.")

    async def section_keyword_recommend(self, data: str, section_per_context:dict):
        # reversed_menu_dict = {value: key for key, value in menu.items()}
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        당신은 웹사이트 랜딩 페이지의 각 섹션에 들어갈 이미지를 검색하는 전문 디자이너입니다.
        구성된 랜딩 페이지의 각 섹션 구성에 알맞게 입력 데이터({{data}})와 섹션 구조 및 요약 내용({{section_per_context}})을 고려하여 이미지를 검색할 영문 키워드를 2개 고르세요.

        ### 규칙
        1. **섹션 구조 및 요약 내용**은 다음과 같은 값을 가집니다:
        {{section_per_context}}
        2. 내용이 부족할 경우, 입력 데이터({{data}})를 함께 고려하여 1개의 단어로 이루어진 영문 키워드를 2개만 선정하세요.
         - 이미지를 잘 찾을 수 있게 섹션별로 1개의 단어로 이루어진 영문 키워드를 2개씩 선정해줘.
         - 각 키워드는 1개의 단어로 이뤄져있어. Ex)Flower
         - 오탈자가 없게 작성해줘.
        3. **JSON 형식 이외의** 어떤 설명, 문장, 주석, 코드 블록도 작성하지 마세요.
        4. 최종 출력은 반드시 **오직 JSON 구조**만 반환해야 합니다.

        ### 출력 형식
        다음 예시처럼 `keyword_structure` 객체를 만들어, 각 섹션을 순서대로 키로 하고 값에 요약 데이터를 채워 넣어 주세요.
        - 예시:
                keyword_structure : {{
                    "Keyword_Hero": "1개의 단어로 이루어진 키워드 2개 작성",
                    "Keyword_Feature": "1개의 단어로 이루어진 키워드 2개 작성",
                    "Keyword_Content": "1개의 단어로 이루어진 키워드 2개 작성",
                    "Keyword_Testimonial": "1개의 단어로 이루어진 키워드 2개 작성",
                    "Keyword_CTA": "1개의 단어로 이루어진 키워드 2개 작성",
                    "Keyword_Pricing": "1개의 단어로 이루어진 키워드 2개 작성",
                    "Keyword_Contact": "1개의 단어로 이루어진 키워드 2개 작성",
                    "Keyword_Footers": "1개의 단어로 이루어진 키워드 2개 작성"
                }}

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {data}
        
        섹션 구조 및 요약 내용:
        {section_per_context}
        <|start_header_id|>assistant<|end_header_id|>
        반드시 **json** 형태로만 결과를 반환
        """
        print(f"prompt len : {len(prompt)} / {prompt}")
        keyword_data = await self.send_request(prompt=prompt)
        print("===============================================")
        print("Let's see keyword_data : ", keyword_data)
        return keyword_data


    async def section_keyword_create_logic(self, data: str, section_per_context: dict):
        """
        data, summary, section을 이용해서 keyword를 생성하는 로직.
        """

        try:
            
            section_context = await self.section_keyword_recommend(data, section_per_context)
            
            # JSON 데이터 파싱
            section_data_with_keyword = await self.process_menu_data(section_context)
            print(f"type of section_context :" , type(section_data_with_keyword))
            print(f"section_context : {section_data_with_keyword}")


            return section_data_with_keyword
        except Exception as e:  # 모든 예외를 잡고 싶다면
            print(f"Error processing landing structure: {e}")

            # menu_data = await self.section_recommend(data)

            # 실패 시 처리
            return  section_per_context
                