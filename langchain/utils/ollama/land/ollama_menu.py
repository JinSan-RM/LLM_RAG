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
        
    async def menu_recommend(self, data: str):
        data = self.clean_data(data)
        print(f"data: {len(data)} / {data}")
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        - 당신은 웹사이트 랜딩 페이지를 구성하는 전문 디자이너입니다.
        - 주어진 섹션 이름과 가중치를 기반으로, 랜딩 페이지를 구성할 섹션 조합을 만드세요.
        - 구성 규칙은 다음과 같습니다:

        1. **필수 섹션**:
        - "Navbars" (가중치: 0)는 항상 포함됩니다.
        - "Hero Header Sections" 또는 "Header Sections" (가중치: 10)는 반드시 포함됩니다.
        - "Footers" (가중치: 90)는 항상 마지막 섹션으로 포함됩니다.

        2. **가중치 규칙**:
        - 가중치가 낮을수록 자주 선택되어야 합니다.
        - 가중치가 높은 섹션은 채택 확률이 낮아야 합니다.


        3. **섹션 목록**:
        아래는 섹션 이름과 가중치입니다:
        - Navbars: 0
        - Hero Header Sections: 10
        - Header Sections: 10
        - Feature Sections: 30
        - CTA Sections: 30
        - Contact Sections: 30
        - Pricing Sections: 30
        - Content Sections: 50
        - Testimonial Sections: 60
        - FAQ Sections: 60
        - Logo Sections: 60
        - Team Sections: 60
        - Gallery Sections: 60
        - Bento Grids: 60
        - Multi-step Forms: 60
        - Comparison Sections: 60
        - Footers: 90
        - Contact Modals: 90

        4. **출력 형식**:

        - 섹션 이름을 순서대로 나열한 JSON 형식으로 작성하세요.
        - 예시:
                menu_structure : {{
                    "1": "Navbars",
                    "2": "Hero Header Sections",
                    "3": "Feature Sections",
                    "4": "Content Sections",
                    "5": "Testimonial Sections",
                    "6": "CTA Sections",
                    "7": "Pricing Sections",
                    "8": "Contact Sections",
                    "9": "Footers"
                }}



        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {data}
        <|start_header_id|>assistant<|end_header_id|>
        - 출력 형식의 예시의 데이터만 출력하세요.
        - 각 조합은 입력 데이터 랜딩페이지지의 목적에 따라 논리적이어야 합니다.
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

    async def menu_create_logic(self, data: str):
        """
        메뉴 데이터를 생성하고 Pydantic 모델을 사용해 처리하는 로직.
        """
        # menu_recommend 실행
        menu_data = await self.menu_recommend(data)
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
                
                # 성공적으로 처리된 menu_structure 반환
                return pydantic_menu_data
            except Exception as e:  # 모든 예외를 잡고 싶다면
                print(f"Error processing landing structure: {e}")
                repeat_count += 1
                menu_data = await self.menu_recommend(data)

        # 실패 시 처리
        print("Failed to process menu data after 3 attempts.")
        return data
                