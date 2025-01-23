from pydantic import BaseModel, TypeAdapter
from config.config import OLLAMA_API_URL
import requests, json, re
from typing import Dict, Union, List

class MenuDict(BaseModel):
    # 루트 모델 대신, 필드 이름을 하나 둔다
    menu_dict: Dict[str, str]

class MenuDataDict(BaseModel):
    menu_structure: Dict[str, str]

class MenuDataList(BaseModel):
    menu_structure: List[str]

class SectionContext(BaseModel):
    section : Dict
    
MenuUnion = Union[MenuDict, MenuDataDict, MenuDataList]

# TypeAdapter로 감싸준다
menu_union_adapter = TypeAdapter(MenuUnion)

class OllamaMenuClient:
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
    
    def parse_menu_data_union(self, data: dict) -> Dict[str, str]:
        # 0) 만약 data가 비어 있다면, 원하는 기본값 / 빈 dict 등으로 처리
        if not data:  # 즉, {}나 None, 혹은 빈 상태
            print("Received an empty dictionary. Returning empty result.")
            return {}

        # 1) 래핑 로직(필요 시)
        if "menu_dict" not in data and "menu_structure" not in data:
            data = {"menu_dict": data}
            
        # TypeAdapter.validate_python() 사용
        parsed = menu_union_adapter.validate_python(data)

        if isinstance(parsed, MenuDict):
            print("menu dict")
            return parsed.menu_dict
        elif isinstance(parsed, MenuDataDict):
            print("menu data")
            return parsed.menu_structure
        elif isinstance(parsed, MenuDataList):
            return {str(i+1): val for i, val in enumerate(parsed.menu_structure)}
        else:
            raise ValueError("Unknown data format")
        
    async def section_recommend(self, data: str):
        data = self.clean_data(data)
        print(f"data: {len(data)} / {data}")
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        - 당신은 웹사이트 랜딩 페이지를 구성하는 전문 디자이너입니다.
        - 주어진 섹션 이름과 가중치를 기반으로, 랜딩 페이지를 구성할 섹션 7~11개의 조합을 만드세요.
        - 구성 규칙은 다음과 같습니다:

        1. **필수 섹션**:
        - "Hero_Header" (가중치: 10)는 반드시 포함됩니다.

        2. **가중치 규칙**:
        - 가중치가 낮을수록 자주 선택되어야 합니다.
        - 가중치가 높은 섹션은 채택 확률이 낮아야 합니다.


        3. **섹션 목록**:
        아래는 섹션 이름과 가중치입니다:
        - Hero_Header : 10
        - Feature : 30
        - CTA : 30
        - Contact : 30
        - Pricing : 30
        - Stats : 30
        - Content : 50
        - Testimonial : 60
        - FAQ : 60
        - Logo : 60
        - Team : 60
        - Gallery : 60
        
        - Timeline : 60
        - Comparison : 60
        - Countdown : 60

       4. **출력 형식**:
        - 반드시 JSON 형식을 사용하여 가중치의 순서에 따라 섹션을 순서대로 나열하세요.
        - 섹션의 개수는 반드시 7~11개 사이여야 합니다.
        - 섹션 이름이 중복되지 않도록 주의하세요.
        - 아래 예시는 단순 참고용입니다. 동일하게 답변할 필요는 없으며, 자유롭게 구성하되 **JSON** 구조만 유지하면 됩니다.

        예시(단순 참고용):
            "menu_structure": {{
                "1": "Hero_Header",
                "2": "Feature",
                "3": "Content",
                "4": "Testimonial",
                "5": "CTA",
                "6": "Pricing",
                "7": "Contact",
            }}
        

        5. **추가 조건**:
        - 각 조합은 입력 데이터(랜딩 페이지 목적, 타깃, 콘셉트 등)에 따라 **논리적**이어야 합니다.
        - 사용자 경험을 고려하여 **주요 섹션(Feature, CTA, Contact)**은 최소 1회 이상 포함하세요.
        - 출력 예시에 구애받지 않고 출력을 생성해주세요.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {data}
        <|start_header_id|>assistant<|end_header_id|>
        - 위 규칙을 충족하는 섹션 배열을 JSON 형태로 구성해주세요.
        - 각 조합은 입력 데이터 랜딩페이지의 목적에 따라 논리적이어야 합니다.
        - 섹션 이름이 중복되지 않도록 주의하세요.
        - 사용자 경험을 고려하여 주요 섹션(Feature, CTA, Contact)은 최소 1회 포함하세요.
        """
        print(f"prompt len : {len(prompt)} / {prompt}")
        menu_data = await self.send_request(prompt=prompt)
        return menu_data
    
    def clean_data(self, text):
        # 고정된 헤더 문자열들
        headers_to_remove = [
            "<|start_header_id|>system<|end_header_id|>",
            "<|start_header_id|>", "<|end_header_id|>",
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|eot_id|>"
        ]
        
        
        # 각 헤더 문자열 제거
        cleaned_text = text
        for header in headers_to_remove:
            cleaned_text = cleaned_text.replace(header, '')
        
        pattern = r'<\|.*?\|>'
        cleaned_text = re.sub(pattern, '', cleaned_text)
    
        return cleaned_text
    
    async def section_per_context(self, data: str, menu:dict):
        reversed_menu_dict = {value: key for key, value in menu.items()}
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        당신은 웹사이트 랜딩 페이지를 구성하는 전문 디자이너입니다.
        1번 프로세스) 구성된 랜딩 페이지의 각 섹션 구성에 알맞게 입력 데이터({{data}})를 요약/분배하여 섹션 내용에 맞춰 작성하세요.
        2번 프로세스) unsplash, pixabey와 같은 이미지 검색 사이트에서 이미지가 잘 검색될 키워드 2개를 선정하여 작성하세요.

        ### 1번 프로세스) 규칙
        1. **섹션 구조**는 다음과 같이 메뉴 이름(키) 목록을 가집니다:
        {menu}
        2. 입력 데이터({{data}})에서 필요한 내용을 발췌하여 각 섹션에 알맞게 배정하세요.
         - 각 섹션마다 내용을 풍부하고 내용 전달할 수 있을 만한 양으로 **150자 정도**로 작성해줘.
         - 오탈자가 없게 작성해줘.
        3. **JSON 형식 이외의** 어떤 설명, 문장, 주석, 코드 블록도 작성하지 마세요.
        4. 최종 출력은 반드시 **오직 JSON 구조**만 반환해야 합니다.

        ### 2번 프로세스) 규칙
        1. **섹션 구조**는 다음과 같이 메뉴 이름(키) 목록을 가집니다:
        {menu}
        2. 아래의 키워드를 추출하는 이유는 이미지 검색 사이트에서 이미지가 잘 검색될 키워드를 찾는 것을 잊지마.
        3. 입력 데이터({{data}})를 고려하여 이미지 검색에 유리한 키워드를 1개 지정해줘.
         - 각 키워드는 최대 3개의 단어로 이루어져있어.
         - 여기서 선정된 키워드는 모든 메뉴에 공통적으로 들어갈거야.
         - 오탈자가 없게 작성해줘.
        4. 1번 프로세스)의 결과물({{menu_structure}})을 참고해서 이미지 검색에 유리한 키워드를 1개 선정해줘.
         - 각 키워드는 최대 3개의 단어로 이루어져있어.
         - 여기서 선정된 키워드는 각각의 메뉴에만 들어갈거야.
         - 오탈자가 없게 작성해줘.
        5. **JSON 형식 이외의** 어떤 설명, 문장, 주석, 코드 블록도 작성하지 마세요.
        6. 최종 출력은 반드시 **오직 JSON 구조**만 반환해야 합니다.

        ### 1번, 2번 프로세스) 출력 형식
        다음 예시처럼 1번 프로세스), 2번 프로세스)를 담는 객체를 만들어, 각 섹션을 순서대로 키로 하고 값에 요약 데이터를 채워 넣어 주세요.
        **아래는 예시이므로 절대 그대로 사용하지말고 프로세스 규칙을 준수하여 작성하세요**
        - 예시_정보통신사업:

                keyword_structure : {{
                    "Hero": {
                        "content": "요약 데이터를 토대로 내용 작성",
                        "keywords": "정보통신, 금융"
                        },
                    "Feature": {
                        "content": "요약 데이터를 토대로 내용 작성",
                        "keywords": "정보통신, 핀테크"
                        },
                    "Content": {
                        "content": "요약 데이터를 토대로 내용 작성",
                        "keywords": "정보통신, 사람"
                        },
                    "Testimonial": {
                        "content": "요약 데이터를 토대로 내용 작성",
                        "keywords": "정보통신, 돈"
                        },
                    "CTA": {
                        "content": "요약 데이터를 토대로 내용 작성",
                        "keywords": "정보통신, 은행"
                        },
                    "Pricing": {
                        "content": "요약 데이터를 토대로 내용 작성",
                        "keywords": "정보통신, 가격"
                        },
                    "Contact": {
                        "content": "요약 데이터를 토대로 내용 작성",
                        "keywords": "정보통신, 전화"
                        }
                }}

        
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {data}
        
        섹션 구조:
        {reversed_menu_dict}
        
        <|start_header_id|>assistant<|end_header_id|>
        반드시 **json** 형태로만 결과를 반환
        """
        print(f"prompt len : {len(prompt)} / {prompt}")
        print("====================================")
        print("It worked!!")
        menu_data = await self.send_request(prompt=prompt)
        
        print("Where is keyword print!! : ", menu_data)
        return menu_data

    async def section_structure_create_logic(self, data: str):
        """
        메뉴 데이터를 생성하고 Pydantic 모델을 사용해 처리하는 로직.
        """
        # menu_recommend 실행
        menu_data = await self.section_recommend(data)
        print(f"menu_data : {menu_data}")
        repeat_count = 0

        while repeat_count < 3:  # 최대 3회 반복
            try:
                # JSON 데이터 파싱
                menu_dict = await self.process_menu_data(menu_data)
                print(f"menu_dict : {menu_dict}")
                
                # Pydantic 모델 생성
                pydantic_menu_data = self.parse_menu_data_union(menu_dict)
                # print(f"pydantic_menu_data : {pydantic_menu_data}")
                
                section_context = await self.section_per_context(data, pydantic_menu_data)
                section_data = await self.process_menu_data(section_context)
                print(f"section_context : {section_context}")


                return pydantic_menu_data, section_data
            except Exception as e:  # 모든 예외를 잡고 싶다면
                print(f"Error processing landing structure: {e}")
                repeat_count += 1
                menu_data = await self.section_recommend(data)

        # 실패 시 처리
        return menu_data, data
                