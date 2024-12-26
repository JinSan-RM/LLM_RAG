from config.config import OLLAMA_API_URL
import requests, json, re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class BaseSection(BaseModel):
    section_type: str

class FeatureItem(BaseModel):
    sub_title: Optional[str] = Field(None, alias="sub_title")
    strength_title: Optional[str] = Field(None, alias="strength_title")
    description: Optional[str] = Field(None, alias="description")
    
class Section(BaseSection):
    main_title: Optional[str] = Field(None, alias="main_title")
    sub_title: Optional[str] = Field(None, alias="sub_title")
    strength_title: Optional[str] = Field(None, alias="strength_title")
    description: Optional[str] = Field(None, alias="description")
    features: Optional[List[FeatureItem]] = Field(None, alias="features")

    class Config:
        allow_population_by_field_name = True
        orm_mode = True
        json_encoders = {
            type(None): lambda v: v  # 기본적으로 None은 제외됨
        }
        
parser = PydanticOutputParser(pydantic_object=Section)

class OllamaLandingClient:
    
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.8, model:str = ''):
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
        
    async def process_data(self, data: str) -> dict:
        """
        LLM의 응답에서 JSON 형식만 추출 및 정리
        """
        try:
            # 1. 줄바꿈 및 공백 정리
            json_text = re.sub(r'\s+', ' ', data).strip()

            # 2. 누락된 쉼표 확인 및 추가
            # 각 키-값 쌍 뒤에 쉼표가 없는 경우 추가
            json_text = re.sub(r'(?<=[}"])\s*(?=")', ', ', json_text)

            # 3. 마지막 쉼표 제거
            # json_text = re.sub(r',\s*([}\]])', r'\1', json_text)

            # 4. JSON 파싱 확인
            parsed_json = json.loads(json_text)  # JSON 파싱
            formatted_json = json.dumps(parsed_json, ensure_ascii=False, indent=4)  # 보기 좋게 포맷팅

            return formatted_json

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {e}")
            print(f"Problematic JSON text: {json_text}")
            raise RuntimeError(f"JSON 파싱 실패: {e}")

    async def generate_section(self, model: str,summary:str, section_name: str):
        """
        랜딩 페이지 섹션을 생성하는 함수
        """
        # 프롬프트 수정 진행중 뒤에 칠판 처럼
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        너는 사이트의 섹션 구조를 정해주고, 그 안에 들어갈 내용을 작성해주는 AI 도우미야.
        다음 **규칙**을 반드시 지켜서, '{section_name}' 섹션에 어울리는 콘텐츠를 생성해줘:

        1) "assistant"처럼 **생성**해야 하고, 규정된 형식을 **절대** 벗어나면 안 된다.
        2) HTML 태그를 다음과 같이 **치환**해서 사용해라:
        - h1  ->  "main_title" (선택)
        - h2  ->  "sub_title"  (선택)
        - h3  ->  "strength_title" (선택)
        - p   ->  "description" (선택)
        - ul  ->  "features" = "[{{ ... }}]" 형태 (예: "[{{ ... }}, {{ ... }}]") (선택)
            - **main_title, sub_title, strength_title, description** 같은 키들은 **필요할 때만** 사용하고, 그 외에는 키를 아예 생성하지 말 것. 
            - 즉, 쓰지 않는 태그(필드)는 **JSON에서 제외**하라.
            - 모든 태그(필드)는 **반드시 문자열**이어야 하며, null이나 배열 형태로 쓰면 안 된다.
            - ul 안에 들어갈 항목(li)도 sub_title, strength_title, description 등의 키를 사용할 수 있음.
            - 단, features 안에서는 "main_title"은 **사용하지 않는다**.
            - **features** 필드만 배열로 사용 가능.
            - **features** 안에 들어가는 각 항목은 `{{}}` 객체 형태를 사용.
            - 여기서도 필요한 키만 사용하고, 필요한 키들은 같은 키들로 features 안의 키들이 동일하게 쓰이게 사용. 안 쓰면 생략.
            - 만약 features 자체가 필요 없다면, **features** 키를 생성하지 말 것.
            - **features** 안의 모든 객체는 동일한 키 구조를 가져야 하며, 동일한 키를 반복해서 사용하지 말 것.
            - 예: 첫 번째 객체에 "sub_title"과 "description" 키가 있다면, 나머지 객체들도 반드시 "sub_title"과 "description"을 사용해야 하며, 추가/생략 불가.
            - features안의 {{}}객체가 4개를 넘을 수 없다.
            - **features 배열**에 들어가는 모든 객체를 섹션 {section_name}의 목적에 맞게 생성하되 **동일한 필드**("sub_title", "strength_title", "description" 중 자유롭게.)를 가질 것을 유의.
            - **features** 배열 내 모든 객체는 동일한 키 세트를 사용해야 하며, 첫 번째 객체에 사용된 키와 동일해야 한다.
            - **features** 배열 내 객체들은 동일한 순서로 키를 배치해 일관성을 유지할 것.
        3) **"section_type"**은 반드시 포함해야 하고, 그 외 태그들은 해당 섹션의 목적과 흐름에 맞춰 **필요한 것만** 사용해도 된다.
        4) **출력은 오직 JSON 형태**로만 해야 하며, 그 외 어떤 설명(문장, 코드, 해설)도 삽입하지 말 것.
        5) 모든 텍스트 내용은 입력 데이터에 맞춰 작성하고, 섹션 '{section_name}'의 목적/흐름을 고려해 자연스럽게 작성한다.
        6) 아래 예시 구조를 준수하되, 필드(태그)들은 섹션에 **필요한 것만** 사용하라.  
            (예: h1이 굳이 필요 없으면 `main_title` 생략 가능)
        7) **출력 형식 예시** (JSON 구조 예시):
        올바른 예시:
            {{
                "section_type": "example",
                "main_title": "메인 제목",
                "description": "설명",
                "features": [
                    {{
                        "sub_title": "부제목1",
                        "description": "설명1"
                    }},
                    {{
                        "sub_title": "부제목2",
                        "description": "설명2"
                    }}
                ]
            }}

            잘못된 예시:
            {{
                "section_type": "example",
                "main_title": "제목",
                "strength_titles": [{{"title": "제목"}}], // 배열 사용 금지
                "descriptions": [{{"desc": "설명"}}], // 배열 사용 금지
                "features": [
                    {{
                        "sub_title": "제목1",
                        "description": "설명1"
                    }},
                    {{
                        "sub_title": "제목2"  // 불일치하는 키 구조 금지
                    }}
                ]
            }}
        
            - **오직 하나의 JSON 객체**만 출력할 것.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            입력 데이터:
            {summary}
            섹션:
            {section_name}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            - 입력데이터를 토대로 키에 해당하는 내용들 채워 JSON만 반환하세요.
            """
        #        7) **출력 형식 예시** (JSON 구조 예시):

        # {{
        #     "section_type": "{section_name}",
        #     "main_title": "string", 
        #     "sub_title": "string",
        #     "description": "string", 
        #     "features": [
        #         {{
        #             "sub_title": "string", 
        #             "strength_title" : "string", 
        #             "description": "string" 
        #         }},
        #         {{
        #             "sub_title": "string", ,
        #             "strength_title" : "string", 
        #             "description": "string" 
        #         }}
        #     ]
        # }}
        repeat_count = 0
        while repeat_count < 3:
            try:
                # 1) LLM에 요청
                raw_json = await self.send_request(prompt=prompt)
                print(f"raw_json : {type(raw_json)} / {raw_json}")

                # 2) JSON 추출 + dict 변환
                p_json = await self.process_data(raw_json)
                print(f"Extracted JSON object: {type(p_json)} / {p_json} ")

                # 3) Pydantic 모델 변환
                parsed_result = Section(**p_json)


                # 4) 성공하면 반환
                return parsed_result

            except RuntimeError as re:
                print(f"Runtime error: {re}")
                repeat_count += 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                repeat_count += 1
        raise RuntimeError("Failed to parse JSON after 3 attempts")