from pydantic import BaseModel, TypeAdapter
from config.config import OLLAMA_API_URL
import requests
import json
import re
from typing import Dict, Union, List
import asyncio
import aiohttp
import ast

class MenuDict(BaseModel):
    # 루트 모델 대신, 필드 이름을 하나 둔다
    menu_dict: Dict[str, str]


class MenuDataDict(BaseModel):
    menu_structure: Dict[str, str]


class MenuDataList(BaseModel):
    menu_structure: List[str]


class SectionContext(BaseModel):
    section: Dict


MenuUnion = Union[MenuDict, MenuDataDict, MenuDataList]

# TypeAdapter로 감싸준다
menu_union_adapter = TypeAdapter(MenuUnion)


class OllamaMenuClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.05, model: str = ''):
        self.api_url = api_url
        self.temperature = temperature
        self.model = model

    async def send_request(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
        }
        # aiohttp ClientSession을 사용하여 비동기 HTTP 요청 수행
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.api_url, json=payload, timeout=40) as response:
                    response.raise_for_status()  # HTTP 에러 발생 시 예외 처리
                    full_response = await response.text()  # 응답을 비동기적으로 읽기
            except aiohttp.ClientError as e:
                print(f"HTTP 요청 실패: {e}")
                raise RuntimeError(f"Ollama API 요청 실패: {e}") from e

        # 전체 응답을 줄 단위로 분할하고 JSON 파싱
        lines = full_response.splitlines()
        all_text = ""
        for line in lines:
            try:
                json_line = json.loads(line.strip())
                all_text += json_line.get("response", "")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue

        return all_text.strip() if all_text else "Empty response received"
    
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
            raise RuntimeError("menu_data의 형식이 올바르지 않습니다.") from e

    def parse_menu_data_union(self, data: dict) -> Dict[str, str]:
        """
        입력 데이터가 아래와 같은 형식이어야 합니다.
        - {"menu_structure": { "1": "Hero", "2": "Feature", ... }}
        - 또는 {"menu_dict": { "1": "Hero", "2": "Feature", ... }}
        - 또는 {"menu_structure": [ "Hero", "Feature", ... ]}
        
        만약 두 키 모두 없으면, data 자체를 {"menu_dict": data}로 래핑합니다.
        그리고 각각의 경우에 따라 적절한 dict를 반환합니다.
        """
        # 0) 데이터가 비어 있다면 빈 dict 반환
        if not data:
            print("Received an empty dictionary. Returning empty result.")
            return {}

        # 1) "menu_dict"와 "menu_structure" 키가 모두 없다면, data를 "menu_dict"로 래핑
        if "menu_dict" not in data and "menu_structure" not in data:
            data = {"menu_dict": data}

        # 2) "menu_dict" 키가 있다면 처리
        if "menu_dict" in data:
            value = data["menu_dict"]
            if isinstance(value, dict):
                print("menu dict")
                return value
            elif isinstance(value, str):
                try:
                    value_parsed = ast.literal_eval(value)
                    if isinstance(value_parsed, dict):
                        print("menu dict (converted from str)")
                        return value_parsed
                    else:
                        raise ValueError("menu_dict 문자열이 dict로 변환되지 않았습니다.")
                except Exception as e:
                    raise ValueError("menu_dict의 값은 dict 타입이어야 합니다.") from e
            else:
                raise ValueError("menu_dict의 값은 dict 타입이어야 합니다.")

        # 3) "menu_structure" 키가 있다면 처리
        elif "menu_structure" in data:
            value = data["menu_structure"]
            if isinstance(value, dict):
                print("menu data")
                return value
            elif isinstance(value, list):
                return {str(i+1): val for i, val in enumerate(value)}
            elif isinstance(value, str):
                try:
                    value_parsed = ast.literal_eval(value)
                    if isinstance(value_parsed, dict):
                        print("menu data (converted from str)")
                        return value_parsed
                    elif isinstance(value_parsed, list):
                        return {str(i+1): val for i, val in enumerate(value_parsed)}
                    else:
                        raise ValueError("menu_structure 문자열이 dict 또는 list로 변환되지 않았습니다.")
                except Exception as e:
                    raise ValueError("menu_structure의 값은 dict 또는 list 타입이어야 합니다.") from e
            else:
                raise ValueError("menu_structure의 값은 dict 또는 list 타입이어야 합니다.")

        else:
            raise ValueError("Unknown data format")
        
    async def section_recommend(self, data: str):
        data = self.clean_data(data)
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        - 당신은 웹사이트 랜딩 페이지를 구성하는 전문 디자이너입니다.
        - 주어진 섹션 이름을 기반으로, 랜딩 페이지를 구성할 섹션 4~6개의 조합을 만드세요.
        - 구성 규칙은 다음과 같습니다:

        1. **필수 섹션**:
        - "Hero"는 반드시 포함됩니다.


        2. **섹션 목록**:
        아래는 섹션 이름입니다.
        - 1번째 섹션 : [Hero]
        - 2번째 섹션 : [Feature, Content]
        - 3번째 섹션 : [CTA, Feature, Content, Gallery, Comparison, Logo]
        - 4번째 섹션 : [Gallery, Comparison, Statistics, Timeline, Countdown, CTA]
        - 5번째 섹션 : [Testimonial, Statistics, Pricing, FAQ, Timeline]
        - 6번째 섹션 : [Contact, FAQ, Logo, Team, Testimonial, Pricing]

       3. **출력 형식**:
        - 반드시 JSON 형식을 사용하여 가중치의 순서에 따라 섹션을 순서대로 나열하세요.
        - 섹션의 개수는 **반드시 4~6개** 사이여야 합니다.
        - 각 섹션의 순서에 따라 알맞은 섹션들을 **섹션 목록의 리스트마다 하나를 자유롭게 선택할 것.**
        - 섹션 이름이 중복되지 않도록 주의하세요.
        - 아래 예시는 단순 참고용입니다. 동일하게 답변할 필요는 없으며, 자유롭게 구성하되 **JSON** 구조만 유지하면 됩니다.
        - 오탈자에 주의하세요.

        예시(단순 참고용):
            "menu_structure": {{
                "1": "section",
                "2": "section",
                "3": "section",
                "4": "section",
                "5": "section",
                "6": "section"
            }}


        5. **추가 조건**:
        - 각 조합은 입력 데이터(랜딩 페이지 목적, 타깃, 콘셉트 등)에 따라 **논리적**이어야 합니다.
        - 출력를 참고해서 다양하게 섹션을 구성해서 출력을 생성해주세요.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {data}
        <|start_header_id|>assistant<|end_header_id|>
        - 위 규칙을 충족하는 섹션 배열을 JSON 형태로 구성해주세요.
        예시(단순 참고용):
            "menu_structure": {{
                "1": "section",
                "2": "section",
                "3": "section",
                "4": "section",
                "5": "section",
                "6": "section"
            }}
        - 섹션의 개수는 6개를 초과해서는 안됩니다.
        - 각 조합은 입력 데이터 랜딩페이지의 목적에 따라 논리적이어야 합니다.
        - 섹션 이름이 중복되지 않도록 주의하세요.
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

    async def section_per_context(self, data: str, menu: dict):
        reversed_menu_dict = {value: key for key, value in menu.items()}
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        당신은 웹사이트 랜딩 페이지를 구성하는 전문 디자이너입니다.
        구성된 랜딩 페이지의 각 섹션 구성에 알맞게 입력 데이터({{data}})를 요약/분배하여 섹션 내용에 맞춰 작성하세요.

        ### 규칙
        1. **섹션 구조**는 다음과 같이 메뉴 이름(키) 목록을 가집니다:
        {menu}
        2. 입력 데이터({{data}})에서 필요한 내용을 발췌하여 각 섹션에 알맞게 배정하세요.
         - 각 섹션마다 내용을 풍부하고 내용 전달할 수 있을 만한 양으로 작성해줘.
         - 오탈자가 없게 작성해줘.
        3. **JSON 형식 이외의** 어떤 설명, 문장, 주석, 코드 블록도 작성하지 마세요.
        4. 최종 출력은 반드시 **오직 JSON 구조**만 반환해야 합니다.
        5. menu_structure는는

        ### 출력 형식
        다음 예시처럼 `menu_structure` 객체를 만들어, 각 섹션을 순서대로 키로 하고 값에 요약 데이터를 채워 넣어 주세요.
        menu_structure의 key는 섹션 구조의 키 값입니다.
        - 예시:
                menu_structure : {{
                    "section": "데이터를 토대로 내용 작성",
                    "section": "데이터를 토대로 내용 작성",
                    "section": "데이터를 토대로 내용 작성",
                    "section": "데이터를 토대로 내용 작성",
                    "section": "데이터를 토대로 내용 작성",
                    "section": "데이터를 토대로 내용 작성"
                }}

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {data}

        섹션 구조:
        {reversed_menu_dict}
        <|start_header_id|>assistant<|end_header_id|>
        반드시 **json** 형태로만 결과를 반환
        - 예시:
                menu_structure : {{
                    "section": "데이터를 토대로 내용 작성",
                    "section": "데이터를 토대로 내용 작성",
                    "section": "데이터를 토대로 내용 작성",
                    "section": "데이터를 토대로 내용 작성",
                    "section": "데이터를 토대로 내용 작성",
                    "section": "데이터를 토대로 내용 작성"
                }}
        """
        menu_data = await self.send_request(prompt=prompt)
        return menu_data

    async def section_structure_create_logic(self, data: str):
        """
        메뉴 데이터를 생성하고 Pydantic 모델을 사용해 처리하는 로직.
        최종적으로 pydantic_menu_data와 section_data가 모두
        {"1": "context", "2": "context", ..., "6": "context"} 형태여야 합니다.
        """
        # menu_recommend 실행
        menu_data = await self.section_recommend(data)
        repeat_count = 0

        while repeat_count < 3:
            try:
                # 1. 메뉴 데이터 처리: JSON 파싱 및 자동 단일 단어 변환
                menu_dict = await self.process_menu_data(menu_data)
                # 자동 변환: 각 값에서 쉼표가 있을 경우 첫 번째 단어만 남기도록 함
                menu_dict = self.transform_to_single_word(menu_dict)
                # if not self.validate_single_word_values(menu_dict):
                #     raise ValueError("메뉴 데이터 값이 단일 단어로 이루어지지 않았습니다.")
                
                # 2. Pydantic 모델 생성
                pydantic_menu_data = self.parse_menu_data_union(menu_dict)
                print(f"parse_menu_data_union pydantic_menu_data : {pydantic_menu_data}")
                
                # 3. 섹션 수 제한 (예: 최대 6개)
                MAX_OTHER_SECTIONS = 6
                other_sections = {k: v for k, v in pydantic_menu_data.items() if k != "Hero"}
                if len(other_sections) > MAX_OTHER_SECTIONS:
                    print("섹션 수가 최대치를 초과했습니다. 초과하는 섹션을 잘라냅니다.")
                    trimmed_other = dict(list(other_sections.items())[:MAX_OTHER_SECTIONS])
                    if "Hero" in pydantic_menu_data:
                        trimmed_other["Hero"] = pydantic_menu_data["Hero"]
                    pydantic_menu_data = trimmed_other
                
                # 4. 메뉴 데이터 내 순수 JSON만 추출 후 단일 단어로 간소화
                # 이미 dict라면 JSON 문자열로 변환한 후 다시 추출합니다.
                pydantic_menu_data = self.extract_menu_structure(json.dumps(pydantic_menu_data))
                print(f"extract_menu_structure pydantic_menu_data : {pydantic_menu_data}")
                pydantic_menu_data = self.simplify_section_structure(pydantic_menu_data)
                print(f"simplify_section_structure pydantic_menu_data : {pydantic_menu_data}")
                
                
                if not self.validate_section_data_structure(pydantic_menu_data):
                    raise ValueError("pydantic_menu_data 형식이 올바르지 않습니다.")
                
                # 5. 섹션 컨텍스트 생성 및 처리
                section_context = await self.section_per_context(data, pydantic_menu_data)
                section_data = await self.process_menu_data(section_context)
                print(f"BF section_data : {section_data}")
                section_data = self.merge_section_fields(section_data)
                print(f"AF section_data : {section_data}")
                
                if not self.validate_section_data_structure(section_data):
                    repeat_count += 1
                    raise ValueError("section_data 형식이 올바르지 않습니다.")
                
                return pydantic_menu_data, section_data

            except Exception as e:
                print(f"Error processing landing structure menu: {e}")
                repeat_count += 1
                menu_data = await self.section_recommend(data)

        return menu_data, data
    
    def simplify_section_structure(self, section_structure) -> dict:
        """
        section_structure가 문자열이면 JSON으로 파싱하고, 
        dict 형태로 변환한 후 각 키의 값이 쉼표(,)로 구분된 경우 첫 번째 단어만 추출하여 반환합니다.
        """
        # 입력값이 문자열이면 JSON 파싱 시도
        if isinstance(section_structure, str):
            try:
                section_structure = json.loads(section_structure)
            except Exception as e:
                raise ValueError(f"section_structure를 JSON으로 파싱할 수 없습니다: {e}")
        # 이제 section_structure는 dict여야 함
        if not isinstance(section_structure, dict):
            raise ValueError("section_structure는 dict 타입이어야 합니다.")

        simplified = {}
        for key, value in section_structure.items():
            if isinstance(value, str):
                simplified[key] = value.split(",")[0].strip()
            else:
                simplified[key] = str(value).strip()
        return simplified
    
    def transform_to_single_word(self, data: dict) -> dict:
        """
        각 value가 쉼표(,)로 구분된 경우, 첫 번째 단어만 추출하여 반환합니다.
        """
        if not isinstance(data, dict):
            raise ValueError("입력 데이터는 dict 타입이어야 합니다.")
        transformed = {}
        for key, value in data.items():
            if isinstance(value, str):
                transformed[key] = value.split(",")[0].strip()
            else:
                transformed[key] = str(value).strip()
        return transformed
    
    def validate_single_word_values(self, data: dict) -> bool:
        """
        각 value가 쉼표(,) 없이 단일 단어로만 이루어져 있는지 검증합니다.
        """
        if not isinstance(data, dict):
            return False
        for key, value in data.items():
            # value가 문자열이 아니거나 쉼표가 포함되어 있으면 False
            if not isinstance(value, str) or ',' in value:
                return False
        return True

    
    def validate_section_data_structure(self, section_data: dict) -> bool:
        """
        section_data가 {"1": "context", "2": "context", ...} 형태인지 검증합니다.
        모든 키가 숫자로 이루어진 문자열인지 확인합니다.
        """
        if not isinstance(section_data, dict):
            return False
        # 모든 키가 숫자로만 구성되어 있는지 확인
        for key in section_data.keys():
            if not isinstance(key, str) or not key.isdigit():
                return False
        return True
    
    def extract_menu_structure(self, text: str) -> dict:
        """
        LLM 응답 문자열에서 코드 블록 내 "menu_structure": { ... } 부분만 추출하여 dict로 반환합니다.
        만약 코드 블록이 없으면, 전체 텍스트에서 JSON을 파싱합니다.
        """
        # 1. 코드 블록 (```json ... ```) 내의 내용을 추출 (비탐욕적 매칭 사용)
        pattern_code_block = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern_code_block, text, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            # 2. 코드 블록이 없으면, "menu_structure": { ... } 패턴으로 추출 시도
            pattern = r'"menu_structure"\s*:\s*(\{.*\})'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_text = match.group(1)
            else:
                # 3. 만약 패턴이 전혀 없으면 전체 텍스트가 순수 JSON이라 가정하고 파싱 시도
                json_text = text.strip()
        
        if not json_text:
            raise ValueError("추출할 JSON 텍스트가 비어 있습니다.")
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {e}")
            raise ValueError(f"추출된 JSON 형식이 올바르지 않습니다: {e}")

    
    def merge_section_fields(self, section_data: dict) -> dict:
        """
        section_data가 다음과 같이 중첩 구조로 되어 있을 때,
        내부의 모든 문자열 값을 추출하여 하나의 스트링으로 병합합니다.
        
        예시:
        Input: {"section": {"title": "Hello", "description": "World", "extra": {"note": "Additional"}}}
        Output: {"section": "Hello World Additional"}
        """
        merged = {}
        for key, value in section_data.items():
            if isinstance(value, dict) or isinstance(value, list):
                merged_value = self.extract_nested_strings(value)
                merged[key] = merged_value
            else:
                merged[key] = str(value).strip()
        return merged
    
    def extract_nested_strings(self, value) -> str:
        """
        재귀적으로 value 내부의 모든 문자열 값을 추출하여 하나의 문자열로 결합합니다.
        - value가 dict이면, 모든 key의 값을 재귀 호출한 후 공백으로 결합합니다.
        - value가 list이면, 각 요소에 대해 재귀 호출한 후 공백으로 결합합니다.
        - 그 외에는 str()로 변환하여 반환합니다.
        """
        if isinstance(value, dict):
            parts = [self.extract_nested_strings(v) for v in value.values()]
            return " ".join(parts).strip()
        elif isinstance(value, list):
            parts = [self.extract_nested_strings(item) for item in value]
            return " ".join(parts).strip()
        else:
            return str(value).strip()