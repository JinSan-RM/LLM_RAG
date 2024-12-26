from config.config import OLLAMA_API_URL
import requests, json, re
from typing import List, Dict, Optional
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser

class BaseSection(BaseModel):
    section_type: str

class FeatureItem(BaseModel):
    sub_title: Optional[str] = None
    strength_title: Optional[str] = None
    description: Optional[str] = None
    
class Section(BaseSection):
    main_title: Optional[str] = None
    sub_title: Optional[str] = None
    strength_title: Optional[str] = None
    description: Optional[str] = None
    features: Optional[List[FeatureItem]] = None

parser = PydanticOutputParser(pydantic_object=Section)

class OllamaLandingClient:
    
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.25, model:str = ''):
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
            # JSON 부분만 추출 (정규표현식 사용)
            json_match = re.search(r"\{.*\}", data, re.DOTALL)
            if not json_match:
                raise ValueError("JSON 형식을 찾을 수 없습니다.")
            
            # JSON 텍스트 추출
            json_text = json_match.group()

            # 이중 따옴표 문제 수정 (선택적)
            json_text = json_text.replace("'", '"').replace("\n", "").strip()
            
            # JSON 파싱
            json_data = json.loads(json_text)

            # 키 오타 수정: 'desc' 또는 'descritption' -> 'description'
            def fix_keys(d):
                if isinstance(d, dict):
                    new_d = {}
                    for k, v in d.items():
                        if k in ['desc', 'descritption', 'descryption', 'descirtion']:
                            fixed_key = 'description'
                        else:
                            fixed_key = k
                        new_d[fixed_key] = fix_keys(v)
                    return new_d
                elif isinstance(d, list):
                    return [fix_keys(item) for item in d]
                else:
                    return d

            fixed_json = fix_keys(json_data)
            print(f"Fixed JSON data: {fixed_json}")  # 디버깅을 위한 출력
            return fixed_json

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {e}")
            raise RuntimeError("data 형식이 올바르지 않습니다.")
        except ValueError as ve:
            print(f"Value error: {ve}")
            raise RuntimeError("data 형식이 올바르지 않습니다.")

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
        - h1  ->  "main_title"
        - h2  ->  "sub_title"
        - h3  ->  "strength_title"
        - p   ->  "description"
        - ul  ->  "features" = "[{{ ... }}]" 형태 (예: "[{{ ... }}, {{ ... }}]")
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
        3) **"section_type"**은 반드시 포함해야 하고, 그 외 태그들은 해당 섹션의 목적과 흐름에 맞춰 **필요한 것만** 사용해도 된다.
        4) **출력은 오직 JSON 형태**로만 해야 하며, 그 외 어떤 설명(문장, 코드, 해설)도 삽입하지 말 것.
        5) 모든 텍스트 내용은 입력 데이터에 맞춰 작성하고, 섹션 '{section_name}'의 목적/흐름을 고려해 자연스럽게 작성한다.
        6) 아래 예시 구조를 준수하되, 필드(태그)들은 섹션에 **필요한 것만** 사용하라.  
            (예: h1이 굳이 필요 없으면 `main_title` 생략 가능)
        7) **출력 형식 예시** (JSON 구조 예시):

        {{
            "section_type": "{section_name}",
            "main_title": "필요하면 작성",
            "description": "필요하면 작성",
            "features": [
                {{
                    "sub_title": "필요하면 작성",
                    "description": "필요하면 작성"
                }},
                {{
                    "sub_title": "필요하면 작성",
                    "description": "필요하면 작성"
                }}
            ]
        }}

        - 위 예시처럼 **features 배열**에 들어가는 모든 객체를 섹션 {section_name}의 목적에 맞게 생성하되 **동일한 필드**("sub_title", "strength_title", "description" 중 자유롭게.)를 가질 것을 유의.
        - **features** 배열 내 모든 객체는 동일한 키 세트를 사용해야 하며, 첫 번째 객체에 사용된 키와 동일해야 한다.
        - **features** 배열 내 객체들은 동일한 순서로 키를 배치해 일관성을 유지할 것.
        - **오직 하나의 JSON 객체**만 출력할 것.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {summary}
        섹션:
        {section_name}

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        - 위 **출력 형식**을 정확히 지키고, 오직 JSON만 반환하세요.
        """
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