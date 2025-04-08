import json
import re
import asyncio
from src.utils.batch_handler import BatchRequestHandler

from itertools import groupby
import pandas as pd
from typing import Dict, Any
from collections import defaultdict
import random


class OpenAISectionSlicer:
    def __init__(self, batch_handler): # , model="gpt-3.5-turbo"
        self.batch_handler = batch_handler
        # self.model = model
        self.extra_body = {}  # 기본값으로 초기화
    
    def set_extra_body(self, extra_body):
        """extra_body 설정 메서드"""
        self.extra_body = extra_body
        
    async def send_request(self, sys_prompt: str, usr_prompt: str, max_tokens: int = 150, extra_body: dict = None) -> str:
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
    
    async def slice_sub_page_to_sections(self, converted_html_tag: str, max_tokens: int = 150):
    
        sys_prompt = f"""
        You are a professional designer who creates website.
        Slice the HumanMessage to the best fit html tag in the candidate list.
        
        #### INSTRUCTIONS ####
        
        1. THE BASIC RULE IS FOLLOW SYMENTIC TAG. SO "h" tags mean heading and "p" tag means description.
        2. THE ORDER OF TAGS MEAN THE POSITION ON THE WEBSITE.
        3. IN THE "usr_prompt", THERE IS NO LIST STRUCTURE. BUT IF YOU THINK IT INCLUDES LIST STRUCTURE, YOU CAN CHOOSE THE ONE HAVE LIST STRUCTURE.
        4. IF THERE IS SAME PATTERNS IN THE INPUT, THAT WOULD BE LIST STRUCTURE. IT 
        
        #### Example ####
        HumanMessage = "img*1_h4_p_h4_p_h4_p
        
        AIMessage =
        {{
            "Section_1": "img*1_h4_p",
            "Section_2": "h4_p",
            "Section_3": "h4_p"
        }}
        """
        
        usr_prompt = f"""
        {converted_html_tag}
        """
        extra_body = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "Section_1": {"type": "string"},
                    "Section_2": {"type": "string"},
                    "Section_3": {"type": "string"},
                    "Section_4": {"type": "string"},
                    "Section_5": {"type": "string"},
                    "Section_6": {"type": "string"},
                    "Section_7": {"type": "string"},
                    "Section_8": {"type": "string"},
                    "Section_9": {"type": "string"},
                    "Section_10": {"type": "string"}
                },
                "required": ["Section_1"],
                "additionalProperties": False,
                "minProperties": 1,
                "maxProperties": 10
            }
        }


        result = await self.send_request(
            sys_prompt=sys_prompt,
            usr_prompt=usr_prompt,
            max_tokens=max_tokens,
            extra_body=extra_body
            )

        if result.success:
            response = result
            return response
        else:
            print(f"Section structure generation error: {result.error}")
            return ""


class OpenAISectionStructureSelector:
    def __init__(self, batch_handler): # , model="gpt-3.5-turbo"
        self.batch_handler = batch_handler
        # self.model = model
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
        
        
        

class OpenAIhtmltopagecontents:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler
        self.section_slicer = OpenAISectionSlicer(batch_handler)
        self.structure_selector = OpenAISectionStructureSelector(batch_handler)
        self.tag_text_generator = OpenAISectionTextGenerator(batch_handler)
        self.block_dataframe = pd.read_excel("src/openai/modoo/matching_block_data/Modoo_matching_blocks.xlsx", index_col=0)
        
        
    async def generate_sub_page_process(self, req, max_tokens: int = 200):        
        results = []
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
        print("TEST_converted_html_tag : ", converted_html_tag)
        extracted_page_context = await self.extracting_page_context(req)
        print("TEST_extracted_page_context : ", extracted_page_context)
        
        sliced_sections = await self.section_slicer.slice_sub_page_to_sections(converted_html_tag)
        sliced_sections_dict = json.loads(sliced_sections.data['generations'][0][0]['text'].strip())            
        print("TEST_sliced_sections_dict : ", sliced_sections_dict)
        
        # NOTE : 애초에 Section 별로 이상하게 잘림
        splited_section_context = await self.split_html_by_tags(extracted_page_context, sliced_sections_dict)
        print("TEST_splited_section_context : ", splited_section_context)
        
        # NOTE : 끝에가 먼저 나오는 케이스 발견 0,1,2로 나와야 하는데 2,0,1로 나옴
        sumed_section_dict = await self.sum_section_dict(sliced_sections_dict, splited_section_context)
        print("TEST_sumed_section_dict : ", sumed_section_dict)
        
        tasks = [await self.generate_sub_page(converted_section_html_tag, extracted_section_context, max_tokens) for converted_section_html_tag, extracted_section_context in sumed_section_dict.items()]
        for task in tasks:
            try:
                result = task
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing task: {e}")
        return results

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
        
    async def convert_to_wanted_output(self, input_str: str) -> str:
        # 언더바(_) 기준으로 토큰 분리
        tokens = input_str.split('_')
        # 첫 번째 토큰은 메인 헤더 (예: h3)
        header = tokens[0]
        # 나머지 토큰들은 li 그룹으로 묶을 대상
        li_tokens = tokens[1:]
        n = len(li_tokens)
        
        # li 그룹의 패턴(반복되는 최소 단위)을 찾기 위한 변수
        group = None
        group_count = 1
        # 가능한 그룹 길이 i를 1부터 n까지 시도하며 전체 li_tokens가 동일한 그룹으로 반복되는지 확인
        for i in range(1, n + 1):
            if n % i == 0:
                groups = [li_tokens[j:j+i] for j in range(0, n, i)]
                if all(g == groups[0] for g in groups):
                    group = groups[0]
                    group_count = len(groups)
                    break
        # 만약 반복되는 그룹을 찾지 못하면 전체 토큰을 하나의 그룹으로 처리
        if group is None:
            group = li_tokens
            group_count = 1
        
        # 그룹 내 토큰은 '+'로 연결 (예: img*1+h3+p)
        group_str = "+".join(group)
        # li 그룹이 여러개면 뒤에 *반복횟수를 붙임 (예: h3_li(img*1+h3+p)*2)
        if group_count > 1:
            second_output = f"{header}_li({group_str})*{group_count}"
        else:
            second_output = f"{header}_li({group_str})"
        
        return second_output


    # NOTE 250408 : [방식 1] 내용에서 중복태그 병합 : while로 순차적으로 돌아야해서 느림
    async def extracting_page_context(self, section_html: str):
        
        # 모든 <img> 태그를 제거하는 정규표현식
        # pattern_img = r'<img[^>]*>'
        
        pattern_nbsp = r'&nbsp;'
        pattern_enter = r'\n'
        
        # 정규표현식을 사용하여 img 태그 제거
        # remove_img = re.sub(pattern_img, '', section_html)
        remove_nbsp = re.sub(pattern_nbsp, '', section_html)
        remove_enter = re.sub(pattern_enter, '', remove_nbsp)
        
        # 동일 태그들 사이의 </tag>와 <tag> 패턴을 개행문자로 치환
        pattern_tags = r'</(p|h2|h3|h4|h5)>\s*<\1>'
        while re.search(pattern_tags, remove_enter):
            remove_enter = re.sub(pattern_tags, ' ', remove_enter)
        return remove_enter
    
    # NOTE 250408 : [방식 2] 내용에서 중복태그 병합 : 선택적으로 돌려서 최적화 한 것이나, [방식 1] 보다 속도가 더 느림
    #               [방식 3] 문득 생각난 건데, 동일한 태그에서 닫힘태그와 열림태그 사이 값이 5
    # def merge_tag_block(self, tag, html):
    #     # 주어진 태그의 연속된 블록을 찾기 위한 정규식
    #     pattern = re.compile(rf'(<{tag}>.*?</{tag}>\s*)+', re.DOTALL)
        
    #     def replacer(match):
    #         # 연속된 태그 블록의 내용을 병합
    #         contents = match.group(0)
    #         # 태그 내 내용만 추출
    #         inner_pattern = re.compile(rf'<{tag}>(.*?)</{tag}>', re.DOTALL)
    #         contents = inner_pattern.findall(contents)
    #         # 내용들을 공백으로 합침
    #         merged_content = " ".join(content.strip() for content in contents)
    #         return f"<{tag}>{merged_content}</{tag}>"
        
    #     # 정규식으로 연속된 블록을 찾아 병합
    #     return re.sub(pattern, replacer, html)


    # async def merge_conditional_tags(self, html: str) -> str:
    #     """
    #     입력된 HTML 문자열에서 대상 태그가 존재하는 경우에만 그 태그 블록을 병합합니다.
        
    #     대상 태그: <p>, <h2>, <h3>, <h4>, <h5>
    #     """
    #     target_tags = ['p', 'h2', 'h3', 'h4', 'h5']
        
    #     for tag in target_tags:
    #         # 해당 태그가 존재하는지 확인 (예: "<p>"가 HTML 내에 있으면)
    #         if re.search(rf'<{tag}>', html):
    #             html = self.merge_tag_block(tag, html)
    #     return html
    # ===========================================================================

    async def split_html_by_tags(self, extracted_context: str, sliced_html_tags: dict):
        result = {}
        
        # 태그 패턴 정의
        tag_patterns = {
            'img': r'<img>',
            'h2': r'<h2>.*?</h2>',
            'h3': r'<h3>.*?</h3>',
            'h4': r'<h4>.*?</h4>',
            'h5': r'<h5>.*?</h5>',
            'p': r'<p>.*?</p>'
        }
        
        # 모든 태그 매칭
        matches = []
        for tag_name, pattern in tag_patterns.items():
            for match in re.finditer(pattern, extracted_context, re.DOTALL):
                matches.append({
                    'tag': tag_name,
                    'start': match.start(),
                    'end': match.end(),
                    'content': match.group()
                })
        
        # 시작 위치 기준으로 정렬
        matches.sort(key=lambda x: x['start'])
        
        # 태그 리스트 만들기
        all_tags = []
        for match in matches:
            if match['tag'] == 'img':
                all_tags.append('img')
            else:
                all_tags.append(match['tag'])
        
        # 각 섹션별로 태그 패턴 파싱
        section_ranges = {}
        current_pos = 0
        
        for section_name, section_pattern in sliced_html_tags.items():
            # img*1_h4_img*1_p 같은 패턴 파싱
            pattern_parts = section_pattern.split('_')
            expected_tags = []
            
            for part in pattern_parts:
                if '*' in part:
                    tag, count = part.split('*')
                    expected_tags.extend([tag] * int(count))
                else:
                    expected_tags.append(part)
            
            # 예상 태그 수 계산
            expected_count = len(expected_tags)
            
            # 현재 위치부터 예상 태그 수만큼의 태그 찾기
            if current_pos + expected_count <= len(matches):
                section_tags = matches[current_pos:current_pos + expected_count]
                section_ranges[section_name] = {
                    'start': section_tags[0]['start'],
                    'end': section_tags[-1]['end'],
                    'content': extracted_context[section_tags[0]['start']:section_tags[-1]['end']]
                }
                current_pos += expected_count
        
        # 결과 반환
        for section_name, section_range in section_ranges.items():
            result[section_name] = section_range['content']
        
        return result


    async def sum_section_dict(self, dict_A, dict_B):
        # dict_A와 dict_B가 동일한 키를 가지고 있는지 확인
        if set(dict_A.keys()) != set(dict_B.keys()):
            print("경고: dict_A와 dict_B의 키가 일치하지 않습니다.")
            
        # 새 dictionary 생성
        new_dict = {}
        
        # dict_A와 dict_B에 공통으로 있는 키에 대해서만 처리
        for key in set(dict_A.keys()) & set(dict_B.keys()):
            # dict_A의 value가 new_dict의 key가 되고, dict_B의 value가 new_dict의 value가 됨
            new_dict[dict_A[key]] = dict_B[key]
        
        return new_dict


    # NOTE : merged된 데이터가 들어오면서 기존 2개를 합치던 방식이 1개로 바뀜
    async def generate_sub_page(self, converted_section_html_tag: str, extracted_section_context: str, max_tokens: int = 200):

        try:
            section_html_tag_LLM_result = await self.structure_selector.select_section_structure(converted_section_html_tag, max_tokens)
            # print(f"\n section_html_tag_LLM_result before : {section_html_tag_LLM_result} \n")
            
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
            
        # print(f"\n section_html_tag_LLM_result after : {section_html_tag_LLM_result} \n")
        
        
        converted_section_html_tag = json.loads(selected_html_tag)
        
        block_ids = self.block_dataframe[self.block_dataframe["converted_html_tag"] == converted_section_html_tag["selected_tag"]].index.tolist()
        
        choiced_block_id = random.choice(block_ids)
        
        choiced_section_tag_length = self.block_dataframe.loc[choiced_block_id, "tag_length"]

        dict_choiced_section_tag_length = json.loads(choiced_section_tag_length)
        
        kv_extracted_context = {"Content" : extracted_section_context}
        
        
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