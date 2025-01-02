from config.config import OLLAMA_API_URL
import requests, json, re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from utils.ollama.land.ollama_tagmatch import parse_html, extract_body_content_with_regex

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
    
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.7, model:str = ''):
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
        
    async def generate_section(self, model: str,summary:str, section_name: str, section_num: int ):
        """
        랜딩 페이지 섹션을 생성하는 함수
        """
        # 프롬프트 수정 진행중 뒤에 칠판 처럼
        
        # ======================================================
        # 태그를 먼저 생성을 요청하고 치환하는 느낌으로 만든 프롬프트
        # ======================================================
        # prompt = f"""
        # <|start_header_id|>system<|end_header_id|>
        # 너는 웹사이트 섹션 구조를 설계하는 전문가야. 주어진 섹션의 특성과 내용을 분석하여 가장 적합한 HTML 구조를 설계한 후, 이를 JSON으로 변환하는 작업을 수행해.

        # 다음 과정을 따라 '{section_name}' 섹션을 설계해줘:

        # 1) 먼저 섹션의 목적과 내용을 파악하여 필요한 HTML 구조를 설계:
        # 예시) Hero 섹션이라면:
        # <h1>메인 타이틀</h1>
        # <p>핵심 설명</p>

        # 또는 Features 섹션이라면:
        # <h2>섹션 타이틀</h2>
        # <ul>
        #     <li>
        #     <h3>기능 제목</h3>
        #     <p>기능 설명</p>
        #     </li>
        # </ul>


        # 5) 출력 예시:
        # Hero 섹션 분석:
        # HTML:
        # <h1>서비스 소개</h1>
        # <p>최고의 경험을 제공합니다.</p>

        # Features 섹션 분석:
        # HTML:
        # <h2>주요 기능</h2>
        # <ul>
        #     <li>
        #         <h2>빠른 속도</h2>
        #         <h3>효율적인 처리</h3>
        #         <p>최적화된 성능 제공</p>
        #     </li>
        #     <li>
        #         <h2>안전한 보안</h2>
        #         <h3>데이터 보호</h3>
        #         <p>안정적인 환경</p>
        #     </li>
        # </ul>

        # <|eot_id|><|start_header_id|>user<|end_header_id|>
        # 입력 데이터:
        # {summary}
        # 섹션:
        # {section_name}
        # <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        # 반드시 단 하나의의**HTML** 형태로 결과를 반환하세요.
        # """
        # =================================
        # 입력 예시만 주고 LLM에게 맞기는 방식
        # =================================
        # prompt = f"""
        # <|start_header_id|>system<|end_header_id|>
        # 너는 사이트의 섹션 구조를 정해주고, {section_name} 섹션에 필요한 콘텐츠를 작성해주는 AI 도우미야.  
        # 아래 규칙들을 **절대** 지키면서 HTML을 생성해줘. (지시사항과 데이터가 충돌할 경우, **지시사항**을 최우선으로 따른다.)

        # 1) **출력 형식**  
        # - 출력은 오직 **HTML 태그**만 사용해야 하며, 추가 설명이나 주석, 어떠한 부가 설명도 포함해서는 안 된다.  
        # - "h1", "h2", "h3", "p", "ul", "li" 태그만 사용 가능.  
        # - 이외의 어떤 태그도 사용하면 안 된다.

        # 2) **태그 구조 규칙**  
        # - **절대로 "<ul>" 자식에 "<ul>"을 중첩**하여 넣으면 안 된다.  
        # - 만약 입력 데이터에 "<ul>"이 중첩되어 있거나, 여러 단계의 "<ul>" 구조를 보유하고 있어도, **한 레벨**로 병합하거나 **필요 시 요약**하여 단일 "<ul>" 구조로만 표현해야 한다.  
        # - "<li>" 태그 안에서는 "<ul>"을 사용하지 않는다.
        # - "<li>" 태그 안에는 오직 "h2", "h3", "p"만 사용할 수 있다.  
        # - 첫 "<li>" 태그 안에 들어가는 태그(키)를 기준으로, 모든 "<li>" 태그는 동일한 구조를 유지한다.  
        # - 하나의 "<ul>" 태그에는 "<li>"가 **최대 4개까지만** 들어갈 수 있다. (4개를 초과할 경우, 나머지 정보를 요약·병합하거나 생략)
        # - "<ul>" 내 모든 "<li>" 객체는 동일한 **키 세트**(동일한 태그 구조)를 가져야 한다.

        # 섹션 이름과 입력 데이터에 적합한 HTML 형태를 만들어주되, 위 규칙을 위반해서는 안 된다.
        # 최종 출력은 별도의 텍스트 설명이나 JSON, 기타 형식은 넣지 않는다. 오직 HTML만 출력한다.
        
        # <|eot_id|><|start_header_id|>user<|end_header_id|> 
        # 섹션: {section_name}
        
        # 입력 데이터: {summary} 
        

        # <|eot_id|><|start_header_id|>assistant<|end_header_id|>

        # 반드시 **HTML**형태로만 결과를 반환하세요. 
        # """
        prompt=f"""
        <|start_header_id|>system<|end_header_id|>
        너는 창의적이고 전문적인 웹사이트 랜딩페이지 생성 전문가야.

        [★ 목적 / 상황]
        - 랜딩페이지의 “{section_num}”번째 “{section_name}” 섹션을 만들어야 한다.
        - HTML 태그는 h1/h2/h3/p/ul/li만 사용 가능.

        [★ 절대 규칙]
        1) **h1, h2, h3, p, ul, li의 HTML 태그만** 사용  
        2) **아무런 속성도 붙이지 말 것** (class, style, id 등 불가)  
        3) **주석, 설명, 빈줄 등 추가 텍스트** 전부 금지 → 순수 HTML만 생성  
        4) **중첩 구조 절대 금지** (ul 안에 ul 불가 / li 안에 ul 불가)  
        5) **ul 태그는 선택적** (꼭 필요하면 1개만 쓸 수 있음)  
        - 만약 ul을 사용한다면, **단 하나만** 허용  
        - li는 1~5개까지만 쓸 수 있음  
        - li 안에는 h2, h3, p 태그만 가능(ul 절대 불가)  
        - 모든 li 구조는 동일한 태그 세트를 유지해야 함  
        6) h1, h2, h3, p 등을 **ul 밖에서도** 최소 2개 이상 써서, 본문을 다양하게 구성.  
        - 즉, “ul 없이” 서술하는 부분과, “ul을 활용한 부분” 둘 다 고려  
        - 굳이 ul을 쓰지 않아도 괜찮다. (필요 없다면 생략)  

        [★ 구성 / 다양성 지침]
        - “{section_name}” 섹션의 성격에 맞춰 **창의적인 톤앤매너**로 작성  
        - 정보가 많으면, **적절히 통합·간소화**  
        - 반복 문장은 지양, **스토리텔링**·비유·사례 등을 자유롭게 활용  
        - 너무 짧거나 중복된 문장 말고, **약간 풍부**하게 써줘  
        - 전문 용어와 친근한 표현을 **적절히 섞어** 서술  
        - 가능하면 **h1**, **h2**, **h3**, **p**를 **다양한 순서**로 조합해봐  
        - (선택) ul을 하나 사용하고 싶다면, 그 안에 최대 5개의 li.  
        - 예: li 안에 “<h3>~</h3><p>~</p>” 구조 등.  
        - h1 태그는 정말 필요하면 1번만 쓸 수 있음(랜딩 섹션 메인 타이틀로).

        [★ 출력 형식]
        - **위 규칙을 철저히 지켜**서 순수 HTML 구조로만 출력하라.
        - 결과에 ul이 여러 개거나 중첩 ul이 있으면 무효이므로, 절대 생성하지 말 것.
        - 결과에 주석, 문맥 외 설명, 속성, 빈줄이 들어가면 안 됨.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        섹션: {section_name}

        입력 데이터: {summary}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>

        "반드시 HTML 형태로만 결과를 반환" 
        """
        repeat_count = 0
        while repeat_count < 3:
            try:
                # 1) LLM에 요청
                raw_json = await self.send_request(prompt=prompt)
                raw_json = extract_body_content_with_regex(raw_json)
                tag = parse_html(raw_json)
                raw_json = re.sub("\n", "", raw_json)
                print(f"raw_json : {type(raw_json)} / {raw_json}")
                # # 2) JSON 추출 + dict 변환
                # p_json = await self.process_data(raw_json)
                print(f"Extracted JSON object: {type(tag)} / {tag} ")
                
                # if isinstance(p_json, str):
                #     p_json = json.loads(p_json) 

                # # 3) Pydantic 모델 변환
                # parsed_result = Section(**p_json)


                # 4) 성공하면 반환
                return raw_json, tag

            except RuntimeError as r:
                print(f"Runtime error: {r}")
                repeat_count += 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                repeat_count += 1
        raise RuntimeError("Failed to parse JSON after 3 attempts")