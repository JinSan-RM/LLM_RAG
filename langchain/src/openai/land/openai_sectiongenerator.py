import json
import re
import asyncio
from src.utils.batch_handler import BatchRequestHandler


class OpenAISectionStructureGenerator:
    def __init__(self, batch_handler): # , model="gpt-3.5-turbo"
        self.batch_handler = batch_handler
        # self.model = model
        self.extra_body = {}  # 기본값으로 초기화
    
    def set_extra_body(self, extra_body):
        """extra_body 설정 메서드"""
        self.extra_body = extra_body
        
    # async def send_request(self, prompt: str, max_tokens: int = 200) -> str:
    async def send_request(self, sys_prompt: str, usr_prompt: str, max_tokens: int = 200, extra_body: dict = None) -> str:
        if extra_body is None:
            extra_body = self.extra_body
        model = "/usr/local/bin/models/gemma-3-4b-it"
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                # "prompt": prompt,
                "model": model,
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
    
    async def create_section_structure(self, final_summary_data: str, max_tokens: int = 200):

        sys_prompt = f"""
        You are a professional designer who creates website landing page.
        Combine the sections according to the Section style list, Instructions and User input.
        
        #### Section style list ####
        
        - 1 section: ["Hero"]
        - 2 section: ["Feature", "Content"]
        - 3 section: ["CTA", "Feature", "Content", "Comparison"]
        - 4 section: ["Comparison", "Statistics", "Countdown", "CTA"]
        - 5 section: ["Testimonial", "Statistics", "Pricing", "FAQ"]
        - 6 section: ["FAQ", "Team", "Testimonial", "Pricing"]
        
        #### INSTRUCTIONS ####
        
        1. READ THE final summary data FROM USER.
        2. THINK ABOUT WHAT KIND OF COMBINATION IS BEST FIT WITH THE final summary data.
        3. THE NUMBER OF SECTIONS **MUST BE BETWEEN 4 AND 6**. IF YOU'RE IDEA WAS MORE THEN 6, THEN REDUCE IT.
        4. TAKING INTO ACCOUNT THE final summary data CHOOSE ONLY ONE SECTION STYLE FOR EACH SECTION IN THE LIST.
        5. BE CAREFUL ABOUT TYPOS.
        6. ENSURE THAT THE OUTPUT MATHCES THE JSON OUTPUT EXAMPLE BELOW.
        """
        
        usr_prompt = f"""
        {final_summary_data}
        """

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
        
class OpenAISectionContentGenerator:
    def __init__(self, output_language, batch_handler: BatchRequestHandler):
        self.output_language = output_language
        self.batch_handler = batch_handler
    
    async def send_request(self, sys_prompt: str, usr_prompt: str, max_tokens: int = 300) -> str:
    # async def send_request(self, prompt: str, max_tokens: int = 300) -> str:
        model = "/usr/local/bin/models/gemma-3-4b-it"
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                # "prompt": prompt,
                "model": model,
                "sys_prompt": sys_prompt,
                "usr_prompt": usr_prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.1,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=120  # 적절한 타임아웃 값 설정
        )
        return response
    
    # 이부분 코쳐야함
    def create_section_prompt(self, section_name, combined_data):
        # # 입력 언어 감지
        # is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in combined_data)
        # language_instruction = "한국어로 작성하세요." if is_korean else "Write in English."

        sys_prompt = f"""
        [System]
        You are a professional content generator for sections in the website landing pages.
        You know what should be included in the composition of each section.
        Your task is to generate concise, unique content based on combined_data.

        #### INSTRUCTIONS ####
        1. WRITE PLAIN TEXT CONTENT FOR THE 'section_name' SECTION.
        2. WRITE BETWEEN 200 AND 300 CHARACTERS IN FOR THE OUTPUT.
        3. USE ONLY RELEVANT PARTS OF THE USER'S DATA: 'combined_data'.
        4. AVOID REPEATING CONTENT.
        5. DO NOT include ANY structure, tags, headers (e.g., ###, [System], [Response]), or metadata. Output ONLY the raw text.
        
        #### Output Language ####
        **{self.output_language}**
        """
        
        usr_prompt = f"""
        Section_name = {section_name}
        all_usr_data = {combined_data}
        """
        return sys_prompt, usr_prompt
    
    async def generate_section_contents_individually(self, section_structure, combined_data, max_tokens=300):
        # 모든 섹션에 대한 태스크 생성
        tasks = []
        for section_key, section_name in section_structure.items():
            sys_prompt, usr_prompt = self.create_section_prompt(section_name, combined_data)
            tasks.append(self.send_request(sys_prompt=sys_prompt, usr_prompt=usr_prompt, max_tokens=max_tokens))
        
        # 모든 태스크를 병렬로 실행
        results = {}
        responses = await asyncio.gather(*tasks)
        
        # 결과 처리
        for section_key, response in zip(section_structure.keys(), responses):
            section_name = section_structure[section_key]
            if response.success and hasattr(response, 'data'):
                # content = response.data.generations[0][0].text
                content = response.data['generations'][0][0]['text']
                # 콘텐츠 정제
                content = self.clean_content(content)
                results[section_name] = content
        
        return results

    def clean_content(self, content):
        # 불필요한 태그, 형식, 메타데이터 제거
        content = re.sub(r'\[.*?\]|\#\#\#|\n|\t|Output:|Example|\*\*.*?\*\*|Response:|`|Title:|Description:|Content:', '', content).strip()
        content = re.sub(r'.*?content for.*?from the context\.', '', content, flags=re.IGNORECASE)
        content = ' '.join(content.split())
        return content

    # async def generate_section_contents_individually(
    #     self,
    #     create_section: dict,
    #     all_usr_data: str,
    #     max_tokens: int = 300
    # ):
    #     is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in all_usr_data)
    #     language_instruction = "Write in Korean." if is_korean else "Write in English."
    #     results = {}  # 리스트 대신 딕셔너리로 초기화
    #     for section_key, section_name in create_section.items():
    #         print(f"section_name: {section_name}")
    #         prompt = f"""
    #         [System]
    #         You are a professional content generator for website landing pages. Your task is to generate concise, unique content based on user data.

    #         #### INSTRUCTIONS ####
    #         1. Write plain text content (150-200 characters) for the 'section_name' section.
    #         2. Use only relevant parts of the user's data: 'all_usr_data'.
    #         3. Avoid repeating content from other sections.
    #         4. Do NOT include ANY structure, tags, headers (e.g., ###, [System], [Response]), or metadata. Output ONLY the raw text.
    #         5. LOOK AT THE INPUT AND **{language_instruction} LANGUAGE TO THE OUTPUT**.
        
    #         [User_Example]
    #         final summary = "all_usr_data"
    #         section list = "section structure"
    #         [/User_Example]

    #         [Assistant_Example]
    #         section_content : {{
    #         "section_name": "Content that Follow the INSTRUCTIONS"
    #         }}
    #         [/Assistant_Example]

    #         [User]
    #         Section_name = {section_name}
    #         all_usr_data = {all_usr_data}
    #         [/User]
    #         """
    #         request = {
    #             "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
    #             "prompt": prompt,
    #             "max_tokens": max_tokens,
    #             "temperature": 0.7,
    #             "top_p": 0.9
    #         }
    #         response = await self.batch_handler.process_single_request(request, request_id=0)
    #         if response.success:
    #             generated_content = response.data.generations[0][0].text.strip()
    #             import re
    #             clean_content = re.sub(r'\[.*?\]|\#\#\#|\*\*.*?\*\*', '', generated_content).strip()
    #             clean_content = clean_content[:300] if len(clean_content) > 300 else clean_content
    #             results[section_name] = clean_content  # append 대신 딕셔너리에 추가
    #     return results


class OpenAISectionGenerator:
    def __init__(self, output_language, batch_handler: BatchRequestHandler):
        output_language=output_language
        self.batch_handler = batch_handler
        self.structure_generator = OpenAISectionStructureGenerator(batch_handler)
        self.content_generator = OpenAISectionContentGenerator(output_language, batch_handler)
        
    async def generate_landing_page(self, requests, max_tokens: int = 200):
        results = []
        extra_body = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "section_1": {
                        "type": "string",
                        "enum": ["Hero"]
                    },
                    "section_2": {
                        "type": "string",
                        "enum": ["Feature", "Content"]
                    },
                    "section_3": {
                        "type": "string",
                        "enum": ["CTA", "Feature", "Content", "Comparison"]
                    },
                    "section_4": {
                        "type": "string",
                        "enum": ["Comparison", "Statistics", "Countdown", "CTA"]
                    },
                    "section_5": {
                        "type": "string",
                        "enum": ["Testimonial", "Statistics", "Pricing", "FAQ"]
                    },
                    "section_6": {
                        "type": "string",
                        "enum": ["FAQ", "Team", "Testimonial", "Pricing"]
                    }
                },
                "required": ["section_1", "section_2", "section_3", "section_4", "section_5", "section_6"]
            }
        }
        
        self.structure_generator.set_extra_body(extra_body)
        for req in requests:

            section_data = await self.generate_section(req.all_usr_data, max_tokens)
            results.append(section_data)
        return results

    # NOTE : merged된 데이터가 들어오면서 기존 2개를 합치던 방식이 1개로 바뀜
    async def generate_section(self, all_usr_data: str, max_tokens: int = 200):
        combined_data = f"PDF summary data = {all_usr_data}"
        allowed_values = {
            "section_1": ["Hero"],
            "section_2": ["Feature", "Content"],
            "section_3": ["CTA", "Feature", "Content", "Comparison"],
            "section_4": ["Comparison", "Statistics", "CTA"],
            "section_5": ["Testimonial", "Statistics", "Pricing", "FAQ"],
            "section_6": ["FAQ", "Team", "Testimonial", "Pricing"]
        }
        cnt = 0
        import random
        while cnt < 3:
            try:
                section_structure_LLM_result = await self.structure_generator.create_section_structure(combined_data, max_tokens)
                
                # 결과 타입 확인 및 처리
                if not section_structure_LLM_result.success:
                    return None
                    
                # 문자열인지 객체인지 확인하여 처리
                section_structure = section_structure_LLM_result.data['generations'][0][0]['text'].strip()

            except Exception as e:
                print(f"Error in generate_section: {str(e)}")
                return None
                # 예외 처리: 객체 구조가 예상과 다를 경우
            section_structure_LLM_result.data['generations'][0][0]['text'] = self.extract_json(section_structure)
            updated_structure = {}
            # section_structure = section_structure_LLM_result.data.generations[0][0].text
            section_structure = section_structure_LLM_result.data['generations'][0][0]['text']

            updated_structure = {}
            used_values = set()  # 이미 사용된 값을 추적
            if section_structure is None or not isinstance(section_structure, dict):
                section_structure = {}
                for section_key in allowed_values.keys():
                    # 해당 섹션에 대해 허용된 값들 중에서 랜덤으로 하나 선택
                    section_structure[section_key] = random.choice(allowed_values[section_key])
            for section, value in section_structure.items():
                if section in allowed_values:
                    if value not in allowed_values[section]:
                        # 허용되지 않은 값일 경우 해당 섹션의 허용된 값 중 무작위 선택
                        value = random.choice(allowed_values[section])
                    
                    # 이미 사용된 값인지 확인
                    if value not in used_values:
                        # 처음 등장하는 값이면 추가
                        used_values.add(value)
                        updated_structure[section] = value
                    # 이미 사용된 값이면 해당 섹션은 건너뜀 (중복 제거)
            section_structure_LLM_result.data['generations'][0][0]['text'] = updated_structure

            # if not isinstance(section_structure_LLM_result.data.generations[0][0].text, dict) or section_structure_LLM_result.data.generations[0][0].text is None:
            if not isinstance(section_structure_LLM_result.data['generations'][0][0]['text'], dict) or section_structure_LLM_result.data['generations'][0][0]['text'] is None:
                section_structure = {}
                for section_key in allowed_values.keys():
                    # 해당 섹션에 대해 허용된 값들 중에서 랜덤으로 하나 선택
                    section_structure[section_key] = random.choice(allowed_values[section_key])
            else:
                break
        # create_section = section_structure_LLM_result.data.generations[0][0].text
        create_section = section_structure_LLM_result.data['generations'][0][0]['text']
        section_content_data = await self.content_generator.generate_section_contents_individually(
            create_section,
            combined_data,
            max_tokens=300
        )

        return {
            "section_structure": section_structure_LLM_result,  # 그대로 유지
            "section_contents": {
                "success": True,
                "data": {
                    "generations": [
                        [
                            {
                                "text": section_content_data
                            }
                        ]
                    ]
                }
            }
        }
        
    def extract_json(self, text):
        # 가장 바깥쪽의 중괄호 쌍을 찾습니다.
        # print("Herere!!!!! : ", text)
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