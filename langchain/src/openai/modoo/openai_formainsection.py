import json
import re
import asyncio
from src.utils.batch_handler import BatchRequestHandler

from itertools import groupby
import pandas as pd
from typing import Dict, Any
from collections import defaultdict
import random

class OpenAISectionStructureSelector:
    def __init__(self, batch_handler, model="gpt-3.5-turbo"):
        self.batch_handler = batch_handler
        self.model = model
        self.extra_body = {}  # 기본값으로 초기화
    
    def set_extra_body(self, extra_body):
        """extra_body 설정 메서드"""
        self.extra_body = extra_body
        
    async def send_request(self, sys_prompt: str, usr_prompt: str, max_tokens: int = 100, extra_body: dict = None) -> str:
        if extra_body is None:
            extra_body = self.extra_body
            
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                # "prompt": prompt,
                "sys_prompt": sys_prompt,
                "usr_prompt": usr_prompt,
                "extra_body": extra_body,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.1,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=300  # 적절한 타임아웃 값 설정
        )
        return response
    
    async def select_section_structure(self, converted_html_tag: str, max_tokens: int = 100):
    
        sys_prompt = f"""
        You are a professional designer who creates website.
        Choose the best fit html tag in the candidate list.
        
        #### INSTRUCTIONS ####
        
        1. THE BASIC RULE IS FOLLOW SYMENTIC TAG. SO "h" tags mean heading and "p" tag means description.
        2. THE ORDER OF TAGS MEAN THE POSITION ON THE WEBSITE.
        3. IN THE "usr_prompt", THERE IS NO LIST STRUCTURE. BUT IF YOU THINK IT INCLUDES LIST STRUCTURE, YOU CAN CHOOSE THE ONE HAVE LIST STRUCTURE.
        
        """
        
        usr_prompt = f"""
        {converted_html_tag}
        """
        
        # result = await self.send_request(prompt, max_tokens)
        result = await self.send_request(
            sys_prompt=sys_prompt,
            usr_prompt=usr_prompt,
            max_tokens=max_tokens,
            extra_body=self.extra_body
            )

        if result.success:
            response = result
            return response
        else:
            print(f"Section structure generation error: {result.error}")
            return ""
        
        
        
        
class OpenAISectionTextGenerator:
    def __init__(self, batch_handler):
        self.batch_handler = batch_handler

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
                                item_properties[sub_key] = {
                                    "type": "string",
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
                elif isinstance(value, (int, str)):

                    properties[key] = {
                        "type": "string",
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

    async def send_request(self, sys_prompt: str, usr_prompt: str, extra_body, max_tokens: int = 500) -> str:
        
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "sys_prompt": sys_prompt,
                "usr_prompt": usr_prompt,
                "extra_body": extra_body,
                "max_tokens": max_tokens,
                "temperature": 0.5,  # 안정성 우선
                "top_p": 0.5,
                "n": 1,
                "stream": False,
                "logprobs": None,
            }, request_id=0),
            timeout=120  # 타임아웃 설정
        )
        return response

    async def generate_tag_text(self, tag_length: dict, extra_body: str, section_context: dict, max_tokens: int = 1000):
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
            - p: Paragraphs of plain text, only express with narrative sentence.
            - li: List items are separated into li_0, li_1, li_2, etc., but share a context.        
            
            #### Input Format ####
            - Section_context= {{section_context}}
            - json_type_tag_list = {{"tag" : "max_length_CHARACTERS"}}
            
            #### Instructions ####
            0. 출력은 반드시 **한국어**로 해.
            1. The results of each tag should be within each max_length_CHARACTERS. The maximum output of each tag must be within 250 characters. If you follow this well, I will give you a tip.
            2. Format all headings (h tags) as concise phrases rather than complete sentences. For example, use 'Effective Marketing Strategies' instead of 'These are effective marketing strategies.'
            3. p tag must express with narrative sentence. For example, use '위븐은 누구나 직관적으로 웹사이트를 만들 수 있게하며, 전문가도 활용가능한 기능을 제공합니다.' instead of '1. 위븐에선 직관적 웹사이트 제작 가능 2. 전문가도 활용가능한 기능 제공'
            4. READ THE SECTION_CONTEXT AND THE MAXIMUM CHARACTER LENGTH, USE THESE AS THE BASIS FOR GENERATING CONTENT FOR EACH TAG IN JSON_TYPE_TAG_LIST.
            5. FOR EACH KEY IN JSON_TYPE_TAG_LIST, THE VALUE REPRESENTS THE MAXIMUM CHARACTER LENGTH.
            6. When generating text, be mindful of the max_length_CHARACTERS constraint. Plan your response so that it naturally concludes with a complete sentence well before reaching the token limit. Prioritize concise expression and avoid starting new thoughts or sentences if they might be cut off.
            7. ENSURE CONTENT IS CONCISE, RELEVANT.
            8. IF A LIST STRUCTURE EXISTS IN JSON_TYPE_TAG_LIST, IT IS CREATED WITH A SIMILAR FORMAT BUT DIFFERENT CONTENT.
            9. DO NOT REPEAT THE SAME SENTENCES OR PATTERNS, AND WRITE WITH UNIQUE CONTENT.
            10. DO NOT INCLUDE MARKDOWN SYMBOLS, ICON and SECTION KEY.
            11. After drafting your response, verify whether it exceeds the max_length_CHARACTERS constraint. If it does, modify your output to conclude with a natural, complete sentence that falls within the token limit. Avoid truncating mid-sentence or leaving thoughts incomplete.

            #### Input Example ####
            Section_context = "재밋은 AI 솔루션을 기반으로 사용자들에게  단하고 편리하게 웹 사이트를 만들 수 있도록 도와주는 선도 서비스입니다. 기업 '위븐'은 AI 솔루션을 통해 일반인들도 쉽게 접근할 수 있으며, 전문가가 사용해도 무방한 에디터와 스튜디오 서비스를 보유하고 있어서 다방면에 능한 서비스를 갖고 있는 기업입니다."
            json_type_tag_list = 
            {{
                "h1_0": "17",
                "h2_0": "19",
                "p_0": "31",
                "li_0": [
                    {{"h2_0": "15", "p_0": "40"}},
                    {{"h2_0": "15", "p_0": "40"}}
                ],
                "p_1: "70"
            }}

            #### Output Example ####
            {{
                "h1_0": "AI로 간편하게 만드는 웹사이트",
                "h2_0": "누구나 쉽게 활용하는 AI 웹 제작",
                "p_0": "위븐은 다방면에 능한 AI 웹 제작 서비스를 제공합니다.",
                "li_0_0": [
                    {{
                        "h2_0": "AI 웹 제작 혁신",
                        "p_0": "기업 '위븐'의 AI 솔루션은 누구나 직관적으로 웹사이트를 만들 수 있도록 지원합니다."
                    }},
                ],
                "li_0_1": [
                    {{
                        "h2_0": "전문가도 만족하는 기능",
                        "p_0": "초보자는 물론 전문가도 활용 가능한 강력한 에디터와 스튜디오 기능을 제공합니다."
                    }},
                "p_1": "재밋은 AI 기반 웹사이트 제작 솔루션을 제공하는 선도 서비스로, 일반 사용자부터 전문가까지 쉽게 활용할 수 있는 강력한 에디터와 스튜디오 서비스를 갖추고 있습니다."
            }}        

        """
        
        # NOTE : section_context가 많이 들어와서 이를 해결하기 위해서 slicing
        section_context_value = section_context.values()
        str_section_context_value = str(section_context_value)
        
        usr_prompt = f"""
            Section_context= {section_context.keys()}, {str_section_context_value[:300]}
            json_type_tag_list= {tag_length}
        """
        extra_body = self.create_extra_body(tag_length)

        result = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "sys_prompt": sys_prompt,
                "usr_prompt": usr_prompt,
                "extra_body": extra_body,
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "top_p": 0.4,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=120  # 적절한 타임아웃 값 설정
        )
        return result

    async def generate_tag_text_process(
        self,
        tag_length: Dict[str, Any],
        section_context: Dict[str, Any],
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        # section_context에서 실제 컨텍스트 문자열 추출

        extra_body = self.create_extra_body(tag_length=tag_length)
        print("extra_body를 보자아 : ", extra_body)
        
        response = await self.generate_tag_text(
            tag_length=tag_length,
            extra_body=extra_body,
            section_context=section_context,
            max_tokens=max_tokens
        )

        # NOTE : JSON 형식 보정 추가
        try:
            json_string = response.data['generations'][0][0]['text']
            print("json 잘 만들어졌는지 보자아 : ", json_string)
            
            # JSON 문자열 파싱 시도
            try:
                parsed_data = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 에러: {e}, Raw: {json_string}")
                # 보정 로직
                stripped_json = json_string.strip()
                if stripped_json.endswith('"'):
                    # 문자열이 잘린 경우
                    fixed_json = stripped_json + '"}'
                    try:
                        parsed_data = json.loads(fixed_json)
                        print("보정 성공: ", fixed_json)
                    except json.JSONDecodeError:
                        return {'gen_content': {'error': f"JSON 보정 실패: {json_string}"}}
                elif stripped_json.endswith('{'):
                    # 객체 시작만 있는 경우
                    fixed_json = stripped_json + '"p_2": "기본 값"}'
                    try:
                        parsed_data = json.loads(fixed_json)
                        print("보정 성공: ", fixed_json)
                    except json.JSONDecodeError:
                        return {'gen_content': {'error': f"JSON 보정 실패: {json_string}"}}
                else:
                    # 그 외: 단순히 "} 추가
                    fixed_json = stripped_json + '"}'
                    try:
                        parsed_data = json.loads(fixed_json)
                        print("보정 성공 (그 외): ", fixed_json)
                    except json.JSONDecodeError:
                        return {'gen_content': {'error': f"JSON 보정 실패: {json_string}"}}

            # 리스트 그룹핑
            li_groups = defaultdict(list)
            for key, value in parsed_data.items():
                if key.startswith('li_'):
                    group_key = '_'.join(key.split('_')[:2])
                    li_groups[group_key].extend(value)

            # transformed_data 생성
            transformed_data = {key: value for key, value in parsed_data.items() if not key.startswith('li_')}
            transformed_data.update(li_groups)

            # response 업데이트
            response.data['generations'][0][0]['text'] = transformed_data
            print("바뀐 response를 보자아 : ", response.data)
            
            return {
                'gen_content': {
                    'data': {
                        'generations': [[{'text': response.data['generations'][0][0]['text']}]]
                    }
                }
            }
        except Exception as e:
            return {'gen_content': {'error': f"Unexpected error: {str(e)}"}}

        
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
        
        
        

class OpenAIhtmltosectioncontents:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler
        self.structure_selector = OpenAISectionStructureSelector(batch_handler)
        self.tag_text_generator = OpenAISectionTextGenerator(batch_handler)
        self.block_dataframe = pd.read_excel("src/openai/modoo/matching_block_data/Modoo_matching_blocks.xlsx", index_col=0)
        
        
    async def generate_main_section(self, req, max_tokens: int = 200):
        
        extra_body = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "selected_tag": {
                        "type": "string",
                        "enum": self.block_dataframe["converted_html_tag"].astype(str).tolist()
                    }
                },
                "required": ["section"]
            }
        }
        
        self.structure_selector.set_extra_body(extra_body)
        
        
        converted_html_tag = await self.converting_html_tag(req)
        extracted_context = await self.extracting_context(req)
        
        result = await self.generate_section(converted_html_tag, extracted_context, max_tokens)
        return result

    async def converting_html_tag(self, section_html: str):
        
        # 정규 표현식으로 태그 추출
        # NOTE 250331 : Modoo에서 다른 tag가 나오면 여기 추가해줘야 함
        tags = re.findall(r'<(img|h2|h3|h4|h5|p)[^>]*>', section_html)
        
        result = []
        
        # groupby를 사용하여 연속된 같은 태그를 그룹화
        for tag, group in groupby(tags):
            group_list = list(group)
            count = len(group_list)
            
            if tag == 'img':
                result.append(f"{tag}*{count}")
            elif tag in ['h2','h3','h4','h5', 'p']:
                result.append(tag)
            else:
                result.append(tag)
        
        return '_'.join(result)
        

        
    async def extracting_context(self, section_html: str):
        
        # 모든 <img> 태그를 제거하는 정규표현식
        # pattern_img = r'<img[^>]*>'
        pattern_nbsp = r'&nbsp;'
        pattern_enter = r'\n'
        
        # 정규표현식을 사용하여 img 태그 제거
        # remove_img = re.sub(pattern_img, '', section_html)
        remove_nbsp = re.sub(pattern_nbsp, '', section_html)
        result = re.sub(pattern_enter, '', remove_nbsp)
        
        return result

    # NOTE : merged된 데이터가 들어오면서 기존 2개를 합치던 방식이 1개로 바뀜
    async def generate_section(self, converted_html_tag: str, extracted_context: str, max_tokens: int = 200):
        
        

        try:
            section_html_tag_LLM_result = await self.structure_selector.select_section_structure(converted_html_tag, max_tokens)            
            # 결과 타입 확인 및 처리
            if not section_html_tag_LLM_result.success:
                print(f"Section structure generation error: {section_html_tag_LLM_result.error}")
                return None
                
            # 문자열인지 객체인지 확인하여 처리
            selected_html_tag = section_html_tag_LLM_result.data['generations'][0][0]['text'].strip()
            
        except Exception as e:
            print(f"Error in generate_section: {str(e)}")
            return None
            # 예외 처리: 객체 구조가 예상과 다를 경우
       
        converted_section_html_tag = json.loads(selected_html_tag)        
        block_ids = self.block_dataframe[self.block_dataframe["converted_html_tag"] == converted_section_html_tag["selected_tag"]].index.tolist()
        choiced_block_id = random.choice(block_ids)        
        choiced_section_tag_length = self.block_dataframe.loc[choiced_block_id, "tag_length"]
        dict_choiced_section_tag_length = json.loads(choiced_section_tag_length)
        kv_extracted_context = {"Content" : extracted_context}
        tag_text_generator_result = await self.tag_text_generator.generate_tag_text_process(dict_choiced_section_tag_length, kv_extracted_context)
        
        return {
            "block_id": {
                "success": True,
                "data": {
                    "generations": [
                        [
                            {
                                "block_id": choiced_block_id
                            }
                        ]
                    ]
                }
            },  # 그대로 유지
            "content": tag_text_generator_result
        }
