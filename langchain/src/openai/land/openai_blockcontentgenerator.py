import asyncio
from typing import Dict, Any
import re
import json
from src.utils.emmet_parser import EmmetParser

from collections import defaultdict

class OpenAIBlockContentGenerator:
    def __init__(self, batch_handler):
        self.batch_handler = batch_handler
        self.emmet_parser = EmmetParser()

    def convert_tag_length_to_schema(self, tag_length):
        properties = {}
        required = []
        
        if isinstance(tag_length, dict):
            for key, value in tag_length.items():
                # 리스트인 경우 (li_0, li_1 등 처리)
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            item_properties = {}
                            item_required = []
                            new_key = f"{key}_{i}"  # li_0_0, li_0_1, li_1_0 등
                            for sub_key, sub_value in item.items():
                                max_length = round(int(sub_value) * 2.5)  # 넉넉히 설정
                                min_length = int(sub_value) // 2  # 예: 최소 절반
                                item_properties[sub_key] = {
                                    "type": "string",
                                    "maxLength": max_length,
                                    "minLength": min_length  # 최소 길이 추가
                                }
                                item_required.append(sub_key)
                            properties[new_key] = {
                                "type": "array",
                                "items": {
                                        "type": "object",
                                        "properties": item_properties,
                                        "required": item_required
                                    },
                                "minItems": 1,
                                "maxItems": 1
                            }
                            
                            required.append(new_key)
                # 딕셔너리인 경우 (중첩 객체)
                elif isinstance(value, dict):
                    nested_schema = self.convert_tag_length_to_schema(value)
                    properties[key] = {
                        "type": "object",
                        "properties": nested_schema["properties"],
                        "required": nested_schema["required"]
                    }
                    required.append(key)
                # 정수나 문자열인 경우 (최대/최소 길이 지정)
                # 정수나 문자열인 경우 (최대/최소 길이 지정)
                elif isinstance(value, (int, str)):
                    max_length = round(int(value) * 2.5)
                    min_length = int(value) // 2  # 예시로 절반 설정
                    properties[key] = {
                        "type": "string",
                        "maxLength": max_length,
                        "minLength": min_length
                    }
                    required.append(key)
                # 기타 타입
                else:
                    properties[key] = {"type": "string"}
                    required.append(key)
        
        return {
            "properties": properties,
            "required": required
        }

    def create_extra_body(self, tag_length):
        schema = self.convert_tag_length_to_schema(tag_length)
        return {
            "guided_json": {
                "type": "object",
                "properties": schema["properties"],
                "required": schema["required"]
            }
        }
    def extract_json(self, text):
        # 가장 바깥쪽의 중괄호 쌍을 찾습니다.
        text = re.sub(r'[\n\r\\\\/]', '', text, flags=re.DOTALL)
        json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            # Handle case where only opening brace is found
            json_str = re.search(r'\{.*', text, re.DOTALL)
            if json_str:
                json_str = json_str.group() + '}'
            else:
                return None

        # Balance braces if necessary
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try a more lenient parsing approach
            try:
                return json.loads(json_str.replace("'", '"'))
            except json.JSONDecodeError:
                return None

    # async def send_request(self, prompt: str, max_tokens: int = 250) -> str:
    async def send_request(self, sys_prompt: str, usr_prompt: str, extra_body, max_tokens: int = 1000) -> str:
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "sys_prompt": sys_prompt,
                "usr_prompt": usr_prompt,
                "extra_body": extra_body,
                "max_tokens": max_tokens,
                "temperature": 0.3,  # 안정성 우선
                "top_p": 0.5,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=120  # 타임아웃 설정
        )
        return response

    async def generate_tag_structure(self, tag_length: dict, extra_body: str, section_context: dict, max_tokens: int = 1000):
        sys_prompt = f"""
            You are an AI assistant that generates content for semantic tags based on the provided Section_context. 
            When given Section_context and json_type_tag_list, read the Section_context first, then create content for each semantic tag key in the json_type_tag_list and replace its value with generated content. 
            Ensure each tag's content aligns with its purpose while strictly adhering to the maximum character length specified in json_type_tag_list.
            #### Semantic Tag Definitions ####
            - h1: The primary title of the web page, summarizing its main topic or purpose (typically one per page).
            - h2: Major section headings, separating key parts of the page.
            - h3: Subheadings under h2, detailing specific topics within sections.
            - h4: Same with h3.
            - h5: Brief supporting text around h tags, enhancing their meaning.
            - p: Paragraphs of plain text, grouping descriptive content.
            - li: List items are separated into li_0, li_1, li_2, etc., but share a context.
            
            #### Instructions ####
            0. 출력은 반드시 **한국어**로 해.
            1. Format all headings (h tags) as concise phrases rather than complete sentences. For example, use 'Effective Marketing Strategies' instead of 'These are effective marketing strategies.' or 'Customer Satisfaction Improvement Methods' instead of 'We should implement customer satisfaction improvement methods.
            2. READ THE SECTION_CONTEXT AND THE MAXIMUM CHARACTER LENGTH, USE THESE AS THE BASIS FOR GENERATING CONTENT FOR EACH TAG IN JSON_TYPE_TAG_LIST.
            3. 2. FOR EACH KEY IN JSON_TYPE_TAG_LIST, THE VALUE REPRESENTS THE MAXIMUM CHARACTER LENGTH.
            4. When generating text, be mindful of the max_length constraint. Plan your response so that it naturally concludes with a complete sentence well before reaching the token limit. Prioritize concise expression and avoid starting new thoughts or sentences if they might be cut off.
            5. ENSURE CONTENT IS CONCISE, RELEVANT.
            6. IF A LIST STRUCTURE EXISTS IN JSON_TYPE_TAG_LIST, GENERATE MULTIPLE ENTRIES WHILE ENSURING EACH MAINTAINS.
            7. AVOIDS REPETITION ACROSS TAGS.
            8. DO NOT INCLUDE MARKDOWN SYMBOLS and SECTION KEY.
            9. After drafting your response, verify whether it exceeds the max_length constraint. If it does, modify your output to conclude with a natural, complete sentence that falls within the token limit. Avoid truncating mid-sentence or leaving thoughts incomplete.
            
          
        """
        
            #         [USER_EXAMPLE]
            # Section_context = "재밋은 AI 솔루션을 기반으로 사용자들에게 간단하고 편리하게 웹 사이트를 만들 수 있도록 도와주는 선도 서비스입니다. 기업 '위븐'은 AI 솔루션을 통해 일반인들도 쉽게 접근할 수 있으며, 전문가가 사용해도 무방한 에디터와 스튜디오 서비스를 보유하고 있어서 다방면에 능한 서비스를 갖고 있는 기업입니다."
            # json_type_tag_list = 
            # {{
            #     "h1_0": "17",
            #     "h2_0": "19",
            #     "p_0": "31",
            #     "li_0": [
            #         {{"h2_0": "15", "p_0": "40"}},
            #         {{"h2_0": "15", "p_0": "40"}}
            #     ],
            #     "p_1: "70"
            # }}

            # [ASSISTANT_EXAMPLE]
            # {{
            #     "h1_0": "AI로 간편하게 만드는 웹사이트",
            #     "h2_0": "누구나 쉽게 활용하는 AI 웹 제작",
            #     "p_0": "위븐은 다방면에 능한 AI 웹 제작 서비스를 제공합니다.",
            #     "li_0_0": [
            #         {{
            #             "h2_0": "AI 웹 제작 혁신",
            #             "p_0": "기업 '위븐'의 AI 솔루션은 누구나 직관적으로 웹사이트를 만들 수 있도록 지원합니다."
            #         }},
            #     ],
            #     "li_0_1": [
            #         {{
            #             "h2_0": "전문가도 만족하는 기능",
            #             "p_0": "초보자는 물론 전문가도 활용 가능한 강력한 에디터와 스튜디오 기능을 제공합니다."
            #         }},
            #     "p_1": "재밋은 AI 기반 웹사이트 제작 솔루션을 제공하는 선도 서비스로, 일반 사용자부터 전문가까지 쉽게 활용할 수 있는 강력한 에디터와 스튜디오 서비스를 갖추고 있습니다."
            # }}
        
        usr_prompt = f"""
            Section_context: {section_context}
            json_type_tag_list: {tag_length}
        """
        extra_body = self.create_extra_body(tag_length)

        result = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "sys_prompt": sys_prompt,
                "usr_prompt": usr_prompt,
                "extra_body": extra_body,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "top_p": 0.4,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=240  # 적절한 타임아웃 값 설정
        )
        return result

#     async def limited_generate(self, coro, semaphore):
#         async with semaphore:
#             return await coro

    async def generate_content(
        self,
        tag_length: Dict[str, Any],
        section_context: Dict[str, Any],
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        # section_context에서 실제 컨텍스트 문자열 추출

        extra_body = self.create_extra_body(tag_length=tag_length)
        print("extra_body를 보자아 : ", extra_body)
        
        response = await self.generate_tag_structure(
            tag_length=tag_length,
            extra_body=extra_body,
            section_context=section_context,
            max_tokens=max_tokens
        )
        json_string = response.data['generations'][0][0]['text']
        # JSON 문자열 파싱
        print("response를 보자아 : ", response)
        
        parsed_data = json.loads(json_string)

        # combined_li = []
        # for key, value in parsed_data.items():
        #     if key.startswith('li_'):
        #         combined_li.extend(value)

        # # 새로운 데이터 구조 생성
        # transformed_data = {key: value for key, value in parsed_data.items() if not key.startswith('li_')}
        # transformed_data['li_0'] = combined_li

        li_groups = defaultdict(list)
        for key, value in parsed_data.items():
            if key.startswith('li_'):
                group_key = '_'.join(key.split('_')[:2])
                li_groups[group_key].extend(value)

        transformed_data = {key: value for key, value in parsed_data.items() if not key.startswith('li_')}
        transformed_data.update(li_groups)

        response.data['generations'][0][0]['text'] = transformed_data
        # print("바뀐 response를 보자아 : ", response)
        return {'gen_content': {'data': {'generations': [[{'text': response.data['generations'][0][0]['text']}]]}}}

    def assign_content(self, result: Dict[str, Any], content: str, key_path: str):
        """
        key_path를 기반으로 생성된 텍스트를 결과 구조 내에 할당합니다.
        """
        keys = key_path.split('.')
        current = result
        for k in keys[:-1]:
            if k.isdigit():
                current = current[int(k)]
            else:
                current = current[k]
        last_key = keys[-1]
        current[last_key] = content
