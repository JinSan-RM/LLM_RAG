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

            # 2. 역슬래시 처리
            json_text = re.sub(r'\\', '\\\\', json_text)  # 역슬래시 이스케이프
            json_text = json_text.replace(r'\\_', '_')  # 잘못된 \_ 처리

            # 3. 제어 문자 제거
            json_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_text)
            
            # 4. 쉼표 누락 처리
            json_text = re.sub(r'(?<=[}"])\s*(?=")', ', ', json_text)

            # 5. 닫히지 않은 중괄호 처리
            open_braces = json_text.count('{')
            close_braces = json_text.count('}')
            
            if open_braces > close_braces:
                json_text += '}' * (open_braces - close_braces)

            # 6. 닫히지 않은 배열 처리
            open_brackets = json_text.count('[')
            close_brackets = json_text.count(']')

            if open_brackets > close_brackets:
                json_text += ']' * (open_brackets - close_brackets)
            # 4. 키-값 쌍 유효성 검사 및 쉼표 추가
            def ensure_commas(json_text):
                lines = json_text.splitlines()
                fixed_lines = []
                for line in lines:
                    if re.match(r'.+:\s*".+"$', line):  # 키-값 형식 확인
                        fixed_lines.append(line + ',')
                    else:
                        fixed_lines.append(line)
                return '\n'.join(fixed_lines)
            
            json_text = ensure_commas(json_text)

            # 5. JSON 파싱 확인
            parsed_json = json.loads(json_text)  # JSON 파싱
            formatted_json = json.dumps(parsed_json, ensure_ascii=False, indent=4)  # 보기 좋게 포맷팅
            # 키 오타 수정
            def fix_keys(d):
                if isinstance(d, dict):
                    return {
                        'description' if k in ['desc', 'descritption', 'descryption', 'descirtion'] else k: 
                        fix_keys(v) for k, v in d.items()
                    }
                elif isinstance(d, list):
                    return [fix_keys(item) for item in d]
                return d

            fixed_json = fix_keys(formatted_json)
            print(f"Fixed JSON data: {fixed_json}")
            return fixed_json


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
            - **main_title, sub_title, strength_title, description, features** 키들만 사용하고, 그 외에는 키를 아예 생성하지 말 것. 
            - 모든 태그(필드)는 **반드시 문자열**이어야 하며, null이나 배열 형태로 쓰면 안 된다.
            - ul 안에 들어갈 항목(li)도 sub_title, strength_title, description의 키만을 사용할 수 있음.
            - **features** 필드만 배열로 사용.
                예: 첫 번째 객체에 "sub_title"과 "description" 키가 있다면, 나머지 객체들도 반드시 "sub_title"과 "description"을 사용해야 하며, 추가/생략 불가.
            - **features**안의 {{}}객체가 4개를 넘을 수 없다.
            - **features 배열**에 들어가는 모든 객체를 섹션 {section_name}의 목적에 맞게 생성하되 **동일한 필드**("sub_title", "strength_title", "description" 중 자유롭게.)를 가질 것을 유의.
            - **features** 배열 내 모든 객체는 동일한 키 세트를 사용해야 하며, 첫 번째 객체에 사용된 키와 동일해야 한다.
        3) **"section_type"**은 반드시 포함해야 하고, 그 외 태그들은 해당 섹션의 목적과 흐름에 맞춰 **필요한 것만** 사용해도 된다.
        4) **출력은 오직 JSON 형태**로만 해야 하며, 그 외 어떤 설명(문장, 코드, 해설, "\")도 삽입하지 말 것.
        5) 아래 예시 구조를 준수하되, 필드(태그)들은 섹션에 **필요한 것만** 사용하라.  
            (예: h1이 굳이 필요 없으면 `main_title` 생략 가능)
        6) **출력 형식 예시** (JSON 구조 예시):
        올바른 예시:
            {{
                "section_type": "{section_name}",
                "main_title": "메인 제목",
                "sub_title": "부 제목",
                "strength_title": "보충 제목",
                "description": "설명",
                "features": [
                    {{
                        "sub_title": "부제목1",
                        "strength_title": "보충 제목1",
                        "description": "설명1"
                    }},
                    {{
                        "sub_title": "부제목2",
                        "strength_title": "보충 제목2",
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
            - 반드시 **JSON**형태를 완벽히 갖춰 결과를 반환하세요.
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
                
                if isinstance(p_json, str):
                    p_json = json.loads(p_json) 

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