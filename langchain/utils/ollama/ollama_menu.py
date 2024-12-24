from pydantic import BaseModel
from config.config import OLLAMA_API_URL
import requests, json, re

class WebsiteMenuStructure(BaseModel):
    menu_structure: dict
    
class OllamaMenuClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.25, model: str =''):
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
        
    async def menu_recommend(self, data: str):
        print(f"data: {len(data)} / {data}")
        data = self.clean_data(data)
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
        - 동일한 조합에서 동일한 섹션이 중복되지 않아야 합니다.

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

        - 순서에 따라 구성된 섹션 이름을 나열하세요.
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
        - 각 조합은 입력 데이터 랜딩페이지지의 목적에 따라 논리적이어야 합니다.
        - 섹션 이름이 중복되지 않도록 주의하세요.
        - 사용자 경험을 고려하여 주요 섹션(Feature, CTA, Contact)은 최소 1회 포함하세요.
        """
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
        # menu_recommend를 실행한 후 결과를 얻기 위해 await 사용
        menu_data = await self.menu_recommend(data)
        print(f"menu_data : {menu_data}")

        # JSON 데이터 파싱
        try:
            menu_dict = await self.process_menu_data(menu_data)
            pydantic_menu_data = WebsiteMenuStructure(menu_structure=menu_dict)
            print(f"pydantic_menu_data : {pydantic_menu_data}")
            return pydantic_menu_data
        except RuntimeError as e:
            print(f"Error processing landing structure: {e}")