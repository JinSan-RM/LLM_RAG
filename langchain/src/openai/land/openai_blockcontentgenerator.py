import asyncio
from typing import Dict, Any
import re
import json
from src.utils.emmet_parser import EmmetParser


class OpenAIBlockContentGenerator:
    def __init__(self, batch_handler):
        self.batch_handler = batch_handler
        self.emmet_parser = EmmetParser()

    async def send_request(self, prompt: str, max_tokens: int = 1000) -> str:
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "top_p": 0.4,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=240  # 적절한 타임아웃 값 설정
        )
        return response

    async def generate_content(
        self, 
        tag_length: Dict[str, Any], 
        section_context: Dict[str, Any], 
        max_tokens: int = 1000) -> Dict[str, Any]:
        # NOTE 250220 : F.E에서 받은거 형식 넣기
    
        print("Want to see how it looks like tag_length : ", tag_length)
        # html = self.emmet_parser.parse_emmet(next(iter(select_block.values())))
        # print(f"html : {html}")
        # print(f"next(iter(section_context.keys())) : {next(iter(section_context.keys()))}")
        # style = self.emmet_parser.font_size(next(iter(section_context.keys())))
        # print(f"style : {style}")

        # 2. **MATCH THE OUTPUT LANGUAGE TO THE PRIMARY LANGUAGE OF SECTION_CONTEXT (E.G., KOREAN OR ENGLISH).**
        prompt = f"""
        [SYSTEM]
        You are an AI assistant that generates content for semantic tags based on the provided Section_context. 
        When given Section_context and json_type_tag_list, read the Section_context first, then create content for each semantic tag key in the json_type_tag_list and replace its value with generated content. 
        Ensure each tag's content aligns with its purpose while strictly adhering to the maximum character length specified in json_type_tag_list.

        #### Semantic Tag Definitions ####
        - h1: The primary title of the web page, summarizing its main topic or purpose (typically one per page).
        - h2: Major section headings, separating key parts of the page.
        - h3: Subheadings under h2, detailing specific topics within sections.
        - h5: Brief supporting text around h tags, enhancing their meaning.
        - p: Paragraphs of plain text, grouping descriptive content.
        - li: List items, where inner tags (e.g., h2, p) should maintain a consistent structure.

        #### Instructions ####
        1. **READ THE SECTION_CONTEXT AND USE IT AS THE BASIS FOR GENERATING CONTENT FOR EACH TAG IN JSON_TYPE_TAG_LIST.**
        2. **FOR EACH KEY IN JSON_TYPE_TAG_LIST, THE VALUE REPRESENTS THE MAXIMUM CHARACTER LENGTH.**
        - If the value is an integer, generate text that does NOT exceed this length.
        - If text needs to be shortened, keep the core meaning while reducing words.
        - Ensure readability and maintain natural sentence structure.
        3. **PRESERVE THE EXACT JSON STRUCTURE PROVIDED IN JSON_TYPE_TAG_LIST, ONLY REPLACING VALUES WITH GENERATED CONTENT.**
        4. **ENSURE CONTENT IS CONCISE, RELEVANT, AND AVOIDS REPETITION ACROSS TAGS.**
        5. **OUTPUT ONLY THE RESULTING JSON, WITHOUT ADDITIONAL TAGS LIKE [SYSTEM] OR METADATA.**
        6. **IF A LIST STRUCTURE EXISTS IN JSON_TYPE_TAG_LIST, GENERATE MULTIPLE ENTRIES WHILE ENSURING EACH MAINTAINS THE DESIGNATED MAXIMUM CHARACTER LENGTH.**
        7. 출력은 반드시 **한국어**로 해.
        
        [USER_EXAMPLE]
        Section_context = "재밋은 AI 솔루션을 기반으로 사용자들에게 간단하고 편리하게 웹 사이트를 만들 수 있도록 도와주는 선도 서비스입니다. 기업 '위븐'은 AI 솔루션을 통해 일반인들도 쉽게 접근할 수 있으며, 전문가가 사용해도 무방한 에디터와 스튜디오 서비스를 보유하고 있어서 다방면에 능한 서비스를 갖고 있는 기업입니다."
        json_type_tag_list = 
        {{
            "h1_0": "17",
            "h2_0": "19",
            "p_0": "31",
            "li_0": [
                {{"h2_0": "15", "p_0": "40"}},
                {{"h2_1": "15", "p_0": "40"}}
            ],
            "p_1: "70"
        }}

        [ASSISTANT_EXAMPLE]
        {{
            "h1_0": "AI로 간편하게 만드는 웹사이트",
            "h2_0": "누구나 쉽게 활용하는 AI 웹 제작",
            "p_0": "위븐은 다방면에 능한 AI 웹 제작 서비스를 제공합니다.",
            "li_0": [
                {{
                    "h2_0": "AI 웹 제작 혁신",
                    "p_0": "기업 '위븐'의 AI 솔루션은 누구나 직관적으로 웹사이트를 만들 수 있도록 지원합니다."
                }},
                {{
                    "h2_1": "전문가도 만족하는 기능",
                    "p_1": "초보자는 물론 전문가도 활용 가능한 강력한 에디터와 스튜디오 기능을 제공합니다."
                }}
            ],
            "p_1": "재밋은 AI 기반 웹사이트 제작 솔루션을 제공하는 선도 서비스로, 일반 사용자부터 전문가까지 쉽게 활용할 수 있는 강력한 에디터와 스튜디오 서비스를 갖추고 있습니다."
        }}

        [USER]
        Section_context: {section_context}
        json_type_tag_list: {tag_length}
        """

        result = await self.send_request(prompt, max_tokens)

        gen_content = self.emmet_parser.tag_sort(result.data.generations[0][0].text.strip())
        gen_content = self.extract_json(gen_content)
        result.data.generations[0][0].text = gen_content
        # if not self.emmet_parser.validate_html_structure(gen_content, html):
        #     raise ValueError(f"생성된 HTML 구조가 예상과 다릅니다: {gen_content}")
        
        return {
            # NOTE 250220 : "/api/block_select" 결과로 미리 보내므로 필요 없어짐짐
            # 'HTML_Tag': next(iter(select_block.values())),
            # 'Block_id': next(iter(select_block.keys())),
            'gen_content': result
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
    #     response = await asyncio.wait_for(
    #         self.batch_handler.process_single_request({
    #             "prompt": prompt,
    #             "max_tokens": max_tokens,
    #             "temperature": 0.1,  # 안정성 우선으로 낮춤
    #             "top_p": 0.1,
    #             "n": 1,
    #             "stream": False,
    #             "logprobs": None
    #         }, request_id=0),
    #         timeout=120
    #     )
    #     return response.data.generations[0][0].text.strip()

    # async def generate_tag_content(self, tag: str, target_length: int, section_context: str, key_path: str, max_tokens: int = 250) -> str:
    #     # 태그의 인덱스 추출 (예: p_0, p_1 등에서 0, 1 등을 추출)
    #     tag_parts = key_path.split('.')[-1].split('_')
    #     tag_base = tag_parts[0]
    #     tag_index = int(tag_parts[1]) if len(tag_parts) > 1 and tag_parts[1].isdigit() else 0
        
    #     # 같은 유형의 태그가 여러 개 있을 경우 약간 다른 관점 제시
    #     variation_prompt = ""
    #     if tag_index > 0:
    #         variations = [
    #             "이전과 다른 관점에서 설명하세요.",
    #             "다른 측면을 강조하세요.",
    #             "다른 단어 선택과 문장 구조를 사용하세요.",
    #             "동일한 내용이지만 완전히 다른 표현으로 작성하세요.",
    #             "비슷한 내용이지만 초점을 다르게 맞추세요."
    #         ]
    #         variation_prompt = f"\n\n중요: 이 텍스트는 같은 주제의 {tag_index+1}번째 변형입니다. {variations[tag_index % len(variations)]}"

    #     prompt = f"""
    #     [SYSTEM]
    #     당신은 정확한 길이로 텍스트를 생성하는 전문가입니다. 오직 정확히 요청된 길이의 텍스트만 반환하세요.

    #     컨텍스트: "{section_context}"

    #     - 내용은 자연스럽게 끝나야 합니다.{variation_prompt}

    #     다음 사항을 절대 포함하지 마세요:
    #     - 설명, 소개, 코드 블록
    #     - 메타 설명이나 지시사항
    #     - 특수 기호, 마크다운, HTML 태그
    #     - 태그 이름이나 태그 설명

    #     [USER]
    #     '{tag}' 유형의 텍스트를 정확히 {target_length}자(±3자)로 생성하세요. 문장이나 단어가 중간에 잘리지 않게 자연스럽게 끝나야 합니다.
    #     """

    #     content = await self.send_request(prompt, max_tokens)

    #     # 후처리 강화
    #     # 모든 마크다운 기호, 태그, 설명 문구 등 제거
    #     content = re.sub(r'\[.*?\]|\{.*?\}|<.*?>|#|\*|\-|\|', '', content)
    #     # 설명용 문구 제거 시도
    #     content = re.sub(r'내용:|결과:|텍스트:|반환:|생성된 내용:|태그 내용:|순수 텍스트:|여기에|입니다.$', '', content)
    #     # 따옴표 제거
    #     content = re.sub(r'[\'"]', '', content)
    #     content = re.sub(r'[\[\]\{\}\(\)\<\>\#\*\-\|]', '', content)
    #     content = re.sub(r'[\'"`]', '', content)
    #     content = re.sub(r'```[\s\S]*?```|`[\s\S]*?`', '', content)
    #     # 공백 정리
    #     content = ' '.join(content.split())
    #     # 앞뒤 공백 제거
    #     content = content.strip()
        
    #     print(f"Generated content for {tag} at {key_path}: '{content}' ({len(content)} chars)")
    #     # 길이 확인 및 조정
    #     # if len(content) > target_length + 10:

    #     # content = self.validate_content(content)
    #     return content
    
    
    # async def generate_content(
    #     self, 
    #     tag_length: Dict[str, Any], 
    #     section_context: Dict[str, Any], 
    #     max_tokens: int = 250
    # ) -> Dict[str, Any]:
    #     context = next(iter(section_context.values()))
    #     result = self.prepare_result_structure(tag_length)

    #     tasks = []
    #     await self.collect_tasks(result, tasks, context)
        
    #     if tasks:
    #         generated_contents = await asyncio.gather(*[task[0] for task in tasks])
    #         for content, key_path in zip(generated_contents, [task[1] for task in tasks]):
    #             self.assign_content(result, content, key_path)

    #     return {'gen_content': {'data': {'generations': [[{'text': result}]]}}}

    # def prepare_result_structure(self, tag_length: Dict[str, Any]) -> Dict[str, Any]:
    #     result = {}
    #     for key, value in tag_length.items():
    #         if isinstance(value, str) and value.isdigit():
    #             result[key] = int(value)
    #         elif isinstance(value, dict):
    #             result[key] = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in value.items()}
    #         elif isinstance(value, list):
    #             result[key] = [{k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in item.items()} for item in value]
    #         else:
    #             result[key] = value
    #     return result

    # async def collect_tasks(self, target: dict, tasks: list, context: str, key_path: list = []):
    #     for key, value in target.items():
    #         current_path = key_path + [key]
    #         if isinstance(value, int):
    #             tasks.append((self.generate_tag_content(key.split('_')[0], value, context, '.'.join(current_path)), '.'.join(current_path)))
    #         elif isinstance(value, dict):
    #             await self.collect_tasks(value, tasks, context, current_path)
    #         elif isinstance(value, list):
    #             for i, item in enumerate(value):
    #                 await self.collect_tasks(item, tasks, context, current_path + [str(i)])

    # def assign_content(self, result: Dict[str, Any], content: str, key_path: str):
    #     keys = key_path.split('.')
    #     current = result
    #     for k in keys[:-1]:
    #         if k.isdigit():
    #             current = current[int(k)]
    #         else:
    #             current = current[k]
    #     last_key = keys[-1]
    #     current[last_key] = content


# ========================================================================================================================================================
# import asyncio
# from typing import Dict, Any
# import re
# import json
# from src.utils.emmet_parser import EmmetParser

# class OpenAIBlockContentGenerator:
#     def __init__(self, batch_handler, concurrency_limit: int = 10):
#         self.batch_handler = batch_handler
#         self.emmet_parser = EmmetParser()
#         # 동일 조건의 요청에 대한 캐시 (tag, target_length, tag_index, section_context)
#         self.cache = {}
#         self.concurrency_limit = concurrency_limit

#         # 후처리에 사용할 정규식들을 미리 컴파일 (속도 개선 효과)
#         self.re_codeblocks = re.compile(r'```[\s\S]*?```|`[\s\S]*?`')
#         self.re_symbols = re.compile(r'\[.*?\]|\{.*?\}|<.*?>|#|\*|\-|\|')
#         self.re_words = re.compile(r'내용:|결과:|텍스트:|반환:|생성된 내용:|태그 내용:|순수 텍스트:|주제:|부연 설명:|예시:|여기에|h1|h2|h3|h5|p|li|h4|:|입니다\.$')
#         self.re_quotes = re.compile(r'[\'"]')
#         self.re_instruct = re.compile(r'\S+\s*유형의 텍스트를 정확히.*?로 생성하세요\.?')
#         self.re_meta = re.compile(r'^(제목\s*:\s*|h1\s*:\s*|부제목\s*:\s*|\*\s*:\s*)', re.IGNORECASE)
#         self.re_leading_meta = re.compile(r'^(?:제목|h1|h2|h3|h4|h5|부제목|\*)[^:]*:\s*', re.IGNORECASE)
#         self.re_tag_names = re.compile(r'\b(?:제목|h1|h2|h3|h4|h5|부제목|\*|p |li|<p>|(p))\s*:\s*', re.IGNORECASE)
#         self.re_leading_colon = re.compile(r'^[:：]+\s*')



#     async def send_request(self, prompt: str, max_tokens: int = 250) -> str:
#         response = await asyncio.wait_for(
#             self.batch_handler.process_single_request({
#                 "prompt": prompt,
#                 "max_tokens": max_tokens,
#                 "temperature": 0.1,  # 안정성 우선
#                 "top_p": 0.1,
#                 "n": 1,
#                 "stream": False,
#                 "logprobs": None
#             }, request_id=0),
#             timeout=120  # 타임아웃 설정
#         )
#         return response.data.generations[0][0].text.strip()

#     async def generate_tag_content(self, tag: str, target_length: int, section_context: str, key_path: str, max_tokens: int = 250) -> str:
#         # key_path 마지막 부분의 인덱스(예: p_0, p_1 등) 추출하여 변형 요청에 활용
#         tag_parts = key_path.split('.')[-1].split('_')
#         tag_index = int(tag_parts[1]) if len(tag_parts) > 1 and tag_parts[1].isdigit() else 0

#         # 캐싱 키 생성 (동일한 조건이면 캐시된 결과 사용)
#         cache_key = (tag, target_length, tag_index, section_context)
#         if cache_key in self.cache:
#             print(f"Cache hit for {cache_key}")
#             return self.cache[cache_key]

#         # 여러 변형이 필요한 경우 약간 다른 프롬프트 추가
#         variation_prompt = ""
#         if tag_index > 0:
#             variations = [
#                 "PLEASE EXPLAIN FROM A DIFFERENT PERSPECTIVE THAN BEFORE.",
#                 "HIGHLIGHT DIFFERENT ASPECTS",
#                 "USE DIFFERENT WORD CHOICES AND SENTENCE STRUCTURES.",
#                 "WRITE THE SAME CONTENT BUT WITH COMPLETELY DIFFERENT EXPRESSIONS.",
#                 "SIMILAR CONTENT, BUT WITH A DIFFERENT FOCUS."
#             ]
#             variation_prompt = f"\n\n**IMPORTANT**: THIS TEXT IS THE {tag_index+1} VARIATION ON THE SAME TOPIC. {variations[tag_index % len(variations)]}"

#         # 랜딩페이지 전체 흐름과 자연스러운 연결성을 고려한 프롬프트
#         is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in section_context)
#         output_language = "Korean" if is_korean else "English"            
#         prompt = f"""
#         [SYSTEM]
#         You are an AI assistant that generates content for semantic tags based on the provided Section_context. 
#         Ensure each tag's content aligns with its purpose while strictly adhering to the maximum character length that given.
#         Considering the flow and consistency of the overall context, write the text of each tag so that it flows naturally.
#         Use Section_context and INSTRUCTIONS to create content in tag with exactly target_length characters (±5 characters).
#         ** IF YOU FOLLOW THE INSTRUCTIONS, I WILL GIVE YOU TIP **
        
#         #### Semantic Tag Definitions ####
#         - h1: The primary title of the web page, summarizing its main topic or purpose (typically one per page).
#         - h2: Major section headings, separating key parts of the page.
#         - h3: Subheadings under h2, detailing specific topics within sections.
#         - h4: Subheadings under h3, detailing specific topics within sections.
#         - h5: Brief supporting text around h tags, enhancing their meaning.
#         - p: Paragraphs of plain text, grouping descriptive content.
#         - li: List items, where inner tags (e.g., h2, p) should maintain a consistent structure.

#         #### INSTRUCTIONS ####
#         1. USING Section_context, THE CONTENT SHOULD END NATURALLY AND THE SENTENCE SHOULD NOT BE INTERRUPTED IN THE MIDDLE. {variation_prompt}
#         2. ENSURE READABILITY AND MAINTAIN NATURAL SENTENCE STRUCTURE.
#         3. MUST EXCLUDE INTRODUCTIONS, CODE BLOCKS, META DESCRIPTIONS, SPECIAL SYMBOLS, MARKDOWN, HTML TAGS, AND TAG NAMES/DESCRIPTIONS.
#         4. IF THE GENERATED TEXT IS OUTSIDE THE REQUIRED LENGTH RANGE, BE SURE TO GENERATE IT AGAIN.
#         5. Read the user_input carefully. The default language is {output_language}
#         [/SYSTEM]
        
#         [USER_EXAMPLE]
#         Section_context = "재밋은 AI 솔루션을 기반으로 사용자들에게 간단하고 편리하게 웹 사이트를 만들 수 있도록 도와주는 선도 서비스입니다."
#         Sementic_tag = p
#         Target_length = 30
#         [/USER_EXAMPLE]
        
#         [ASSISTANT_EXAMPLE]
#         재밋 AI 솔루션으로 간단하게 웹 사이트를 만들 수 있습니다.
#         [/ASSISTANT_EXAMPLE]
        
#         [USER]
#         Section_context = {section_context}
#         Sementic_tag = {tag}
#         Target_length = {target_length}
#         [/USER]
        
#         """
#         content = await self.send_request(prompt, max_tokens)

#         # 후처리: 미리 컴파일한 정규식들을 순차적으로 적용
#         content = self.re_codeblocks.sub('', content)
#         content = self.re_symbols.sub('', content)
#         content = self.re_words.sub('', content)
#         content = self.re_quotes.sub('', content)
#         content = self.re_instruct.sub('', content)
#         content = self.re_meta.sub('', content)
#         content = self.re_leading_meta.sub('', content)
#         content = self.re_tag_names.sub('', content)
#         content = self.re_leading_colon.sub('', content)
#         content = ' '.join(content.split()).strip()

#         print(f"Generated content for {tag} at {key_path}: '{content}' ({len(content)} chars)")
#         # 캐시에 저장
#         self.cache[cache_key] = content
#         return content

#     async def limited_generate(self, coro, semaphore):
#         async with semaphore:
#             return await coro

#     async def generate_content(
#         self, 
#         tag_length: Dict[str, Any], 
#         section_context: Dict[str, Any], 
#         max_tokens: int = 200
#     ) -> Dict[str, Any]:
#         # section_context에서 실제 컨텍스트 문자열 추출
#         context = next(iter(section_context.values()))
#         result = self.prepare_result_structure(tag_length)

#         tasks = []
#         await self.collect_tasks(result, tasks, context)

#         # 동시 실행 제한을 위해 semaphore 사용
#         semaphore = asyncio.Semaphore(self.concurrency_limit)
#         if tasks:
#             generated_contents = await asyncio.gather(*[
#                 self.limited_generate(task[0], semaphore) for task in tasks
#             ])
#             for content, key_path in zip(generated_contents, [task[1] for task in tasks]):
#                 self.assign_content(result, content, key_path)

#         return {'gen_content': {'data': {'generations': [[{'text': result}]]}}}

#     def prepare_result_structure(self, tag_length: Dict[str, Any]) -> Dict[str, Any]:
#         result = {}
#         for key, value in tag_length.items():
#             if isinstance(value, str) and value.isdigit():
#                 result[key] = int(value)
#             elif isinstance(value, dict):
#                 result[key] = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in value.items()}
#             elif isinstance(value, list):
#                 result[key] = [
#                     {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in item.items()} 
#                     for item in value
#                 ]
#             else:
#                 result[key] = value
#         return result

#     async def collect_tasks(self, target: dict, tasks: list, context: str, key_path: list = []):
#         """
#         재귀적으로 태그 구조를 순회하며 각 태그에 대해 generate_tag_content 요청을 tasks 리스트에 추가합니다.
#         """
#         for key, value in target.items():
#             current_path = key_path + [key]
#             if isinstance(value, int):
#                 tasks.append((self.generate_tag_content(key.split('_')[0], value, context, '.'.join(current_path)), '.'.join(current_path)))
#             elif isinstance(value, dict):
#                 await self.collect_tasks(value, tasks, context, current_path)
#             elif isinstance(value, list):
#                 for i, item in enumerate(value):
#                     await self.collect_tasks(item, tasks, context, current_path + [str(i)])

#     def assign_content(self, result: Dict[str, Any], content: str, key_path: str):
#         """
#         key_path를 기반으로 생성된 텍스트를 결과 구조 내에 할당합니다.
#         """
#         keys = key_path.split('.')
#         current = result
#         for k in keys[:-1]:
#             if k.isdigit():
#                 current = current[int(k)]
#             else:
#                 current = current[k]
#         last_key = keys[-1]
#         current[last_key] = content


# ========================================================================================================================================================
import asyncio
from typing import Dict, Any
import re
import json
from src.utils.emmet_parser import EmmetParser

class OpenAIBlockContentGenerator:
    def __init__(self, batch_handler, concurrency_limit: int = 10):
        self.batch_handler = batch_handler
        self.emmet_parser = EmmetParser()
        # 동일 조건의 요청에 대한 캐시 (tag, target_length, tag_index, section_context)
        self.cache = {}
        self.concurrency_limit = concurrency_limit

        # 후처리에 사용할 정규식들을 미리 컴파일 (속도 개선 효과)
        self.re_codeblocks = re.compile(r'```[\s\S]*?```|`[\s\S]*?`')
        self.re_symbols = re.compile(r'\[.*?\]|\{.*?\}|<.*?>|#|\*|\-|\|')
        self.re_words = re.compile(r'내용:|결과:|텍스트:|반환:|생성된 내용:|태그 내용:|순수 텍스트:|주제:|부연 설명:|예시:|여기에|h1|h2|h3|h5|p|li|h4|:|입니다\.$')
        self.re_quotes = re.compile(r'[\'"]')
        self.re_instruct = re.compile(r'\S+\s*유형의 텍스트를 정확히.*?로 생성하세요\.?')
        self.re_meta = re.compile(r'^(제목\s*:\s*|h1\s*:\s*|부제목\s*:\s*|\*\s*:\s*)', re.IGNORECASE)
        self.re_leading_meta = re.compile(r'^(?:제목|h1|h2|h3|h4|h5|부제목|\*)[^:]*:\s*', re.IGNORECASE)
        self.re_tag_names = re.compile(r'\b(?:제목|h1|h2|h3|h4|h5|부제목|\*|p |li|<p>|(p))\s*:\s*', re.IGNORECASE)
        self.re_leading_colon = re.compile(r'^[:：]+\s*')



    async def send_request(self, prompt: str, max_tokens: int = 250) -> str:
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.3,  # 안정성 우선
                "top_p": 0.5,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=120  # 타임아웃 설정
        )
        return response.data.generations[0][0].text.strip()

    async def generate_tag_content(self, tag: str, target_length: int, section_context: str, key_path: str, max_tokens: int = 250) -> str:
        # key_path 마지막 부분의 인덱스(예: p_0, p_1 등) 추출하여 변형 요청에 활용
        tag_parts = key_path.split('.')[-1].split('_')
        tag_index = int(tag_parts[1]) if len(tag_parts) > 1 and tag_parts[1].isdigit() else 0

        # 캐싱 키 생성 (동일한 조건이면 캐시된 결과 사용)
        cache_key = (tag, target_length, tag_index, section_context)
        if cache_key in self.cache:
            print(f"Cache hit for {cache_key}")
            return self.cache[cache_key]

        # 여러 변형이 필요한 경우 약간 다른 프롬프트 추가
        variation_prompt = ""
        if tag_index > 0:
            variations = [
                "PLEASE EXPLAIN FROM A DIFFERENT PERSPECTIVE THAN BEFORE.",
                "HIGHLIGHT DIFFERENT ASPECTS",
                "USE DIFFERENT WORD CHOICES AND SENTENCE STRUCTURES.",
                "WRITE THE SAME CONTENT BUT WITH COMPLETELY DIFFERENT EXPRESSIONS.",
                "SIMILAR CONTENT, BUT WITH A DIFFERENT FOCUS."
            ]
            variation_prompt = f"\n\n**IMPORTANT**: THIS TEXT IS THE {tag_index+1} VARIATION ON THE SAME TOPIC. {variations[tag_index % len(variations)]}"

        # 랜딩페이지 전체 흐름과 자연스러운 연결성을 고려한 프롬프트
        is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in section_context)
        output_language = "Korean" if is_korean else "English"            
        prompt = f"""
        [SYSTEM]
        You are an AI assistant that generates content for semantic tags based on the provided Section_context. 
        Ensure each tag's content aligns with its purpose while strictly adhering to the maximum character length that given.
        Considering the flow and consistency of the overall context, write the text of each tag so that it flows naturally.
        ** IF YOU FOLLOW THE INSTRUCTIONS, I WILL GIVE YOU TIP **
        
        #### Semantic Tag Definitions ####
        - h1: The primary title of the web page, summarizing its main topic or purpose (typically one per page).
        - h2: Major section headings, separating key parts of the page.
        - h3: Subheadings under h2, detailing specific topics within sections.
        - h4: Subheadings under h3, detailing specific topics within sections.
        - h5: Brief supporting text around h tags, enhancing their meaning.
        - p: Paragraphs of plain text, grouping descriptive content.
        - li: List items, where inner tags (e.g., h2, p) should maintain a consistent structure.

        #### INSTRUCTIONS ####
        1. USING SECTION_CONTEXT, THE CONTENT SHOULD END NATURALLY AND THE SENTENCE SHOULD NOT BE INTERRUPTED IN THE MIDDLE. {variation_prompt}
        2. ENSURE READABILITY AND MAINTAIN NATURAL SENTENCE STRUCTURE.
        3. MUST EXCLUDE INTRODUCTIONS, CODE BLOCKS, META DESCRIPTIONS, SPECIAL SYMBOLS, MARKDOWN, HTML TAGS, AND TAG NAMES/DESCRIPTIONS.
        4. IF THE GENERATED TEXT IS OUTSIDE THE REQUIRED LENGTH RANGE, BE SURE TO GENERATE IT AGAIN.
        5. Read the user_input carefully. The default language is {output_language}
        
        [USER]
        Section_context = {section_context}
        Use Section_context and INSTRUCTIONS to create '{tag}' type text with exactly {target_length} characters (±5 characters).
        """
        content = await self.send_request(prompt, max_tokens)

        # 후처리: 미리 컴파일한 정규식들을 순차적으로 적용
        content = self.re_codeblocks.sub('', content)
        content = self.re_symbols.sub('', content)
        content = self.re_words.sub('', content)
        content = self.re_quotes.sub('', content)
        content = self.re_instruct.sub('', content)
        content = self.re_meta.sub('', content)
        content = self.re_leading_meta.sub('', content)
        content = self.re_tag_names.sub('', content)
        content = self.re_leading_colon.sub('', content)
        content = ' '.join(content.split()).strip()

        # 캐시에 저장
        self.cache[cache_key] = content
        return content

    async def limited_generate(self, coro, semaphore):
        async with semaphore:
            return await coro

    async def generate_content(
        self, 
        tag_length: Dict[str, Any], 
        section_context: Dict[str, Any], 
        max_tokens: int = 200
    ) -> Dict[str, Any]:
        # section_context에서 실제 컨텍스트 문자열 추출
        context = next(iter(section_context.values()))
        result = self.prepare_result_structure(tag_length)

        tasks = []
        await self.collect_tasks(result, tasks, context)

        # 동시 실행 제한을 위해 semaphore 사용
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        if tasks:
            generated_contents = await asyncio.gather(*[
                self.limited_generate(task[0], semaphore) for task in tasks
            ])
            for content, key_path in zip(generated_contents, [task[1] for task in tasks]):
                self.assign_content(result, content, key_path)

        return {'gen_content': {'data': {'generations': [[{'text': result}]]}}}

    def prepare_result_structure(self, tag_length: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in tag_length.items():
            if isinstance(value, str) and value.isdigit():
                result[key] = int(value)
            elif isinstance(value, dict):
                result[key] = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in value.items()}
            elif isinstance(value, list):
                result[key] = [
                    {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in item.items()} 
                    for item in value
                ]
            else:
                result[key] = value
        return result

    async def collect_tasks(self, target: dict, tasks: list, context: str, key_path: list = []):
        """
        재귀적으로 태그 구조를 순회하며 각 태그에 대해 generate_tag_content 요청을 tasks 리스트에 추가합니다.
        """
        for key, value in target.items():
            current_path = key_path + [key]
            if isinstance(value, int):
                tasks.append((self.generate_tag_content(key.split('_')[0], value, context, '.'.join(current_path)), '.'.join(current_path)))
            elif isinstance(value, dict):
                await self.collect_tasks(value, tasks, context, current_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    await self.collect_tasks(item, tasks, context, current_path + [str(i)])

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