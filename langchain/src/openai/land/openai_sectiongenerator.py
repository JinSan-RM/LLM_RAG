import json
import re
import asyncio
from src.utils.batch_handler import BatchRequestHandler


class OpenAISectionStructureGenerator:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler

    async def send_request(self, prompt: str, max_tokens: int = 200) -> str:
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
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
        
        prompt = f"""
        [System]
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
        
        [/System]

        [User_Example]
        final summary data = "final summary data"
        [/User_Example]

        [Assistant_Example]
        "menu_structure": {{
        "section_1": "Hero",
        "section_2": "section style",
        "section_3": "section style",
        "section_4": "section style",
        "section_5": "section style",
        "section_6": "section style"
        }}

        [/Assistant_Example]
        
        [User]
        final summary data = {final_summary_data}
        [/User]
        """

        result = await self.send_request(prompt, max_tokens)
        

        
        if result.success:
            response = result
            print(f"Section structure response: {response}")
            return response
        else:
            print(f"Section structure generation error: {result.error}")
            return ""
        
class OpenAISectionContentGenerator:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler
        
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
    
    async def send_request(self, prompt: str, max_tokens: int = 300) -> str:
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
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
    
    # 이부분 코쳐야함
    def create_section_prompt(self, section_name, combined_data):
        # 입력 언어 감지
        is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in combined_data)
        language_instruction = "한국어로 작성하세요." if is_korean else "Write in English."
        
        prompt = f"""
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
        6. {language_instruction}
        
        [User]
        Section_name = {section_name}
        all_usr_data = {combined_data}
        """
        return prompt
    
    async def generate_section_contents_individually(self, section_structure, combined_data, max_tokens=300):
        # 모든 섹션에 대한 태스크 생성
        tasks = []
        for section_key, section_name in section_structure.items():
            prompt = self.create_section_prompt(section_name, combined_data)
            tasks.append(self.send_request(prompt, max_tokens))
        
        # 모든 태스크를 병렬로 실행
        results = {}
        responses = await asyncio.gather(*tasks)
        
        # 결과 처리
        for section_key, response in zip(section_structure.keys(), responses):
            print("==============================================")
            print("response : ", response)
            print("==============================================")
            section_name = section_structure[section_key]
            if response.success and hasattr(response, 'data'):
                content = response.data.generations[0][0].text
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
        
    # async def create_section_contents(self, all_usr_data: str, structure: dict, max_tokens: int = 1800):        
        
    #     prompt = f"""
    #     [System]
    #     You are a professional planner who organizes the content of your website landing page.
    #     Write in the section content by summarizing and distributing the user's final summary data.

    #     #### INSTRUCTIONS ####
    #     1. CHECK THE FINAL SUMMARY FOR NECESSARY INFORMATION AND ORGANIZE IT APPROPRIATELY FOR EACH SECTION.
    #     2. FOR EACH SECTION, PLEASE WRITE ABOUT 150 TO 200 CHARACTERS SO THAT THE CONTENT IS RICH AND CONVEYS THE CONTENT.
    #     3. WRITE DIFFERENT CONTENT FOR EACH SECTION.
    #     4. WRITE WITHOUT TYPOS.
    #     5. LOOK AT THE INPUT AND **FOLLOW THE INPUT LANGUAGE TO THE OUTPUT**.
    #     6. ENSURE THAT THE OUTPUT MATCHES THE JSON OUTPUT EXAMPLE BELOW.
    #     [/System]

    #     [User_Example]
    #     final summary = "all_usr_data"
    #     section list = "section structure"
    #     [/User_Example]

    #     [Assistant_Example]
    #     section_content : {{
    #     "Hero": "Content that Follow the INSTRUCTIONS",
    #     "section style": "Content that Follow the INSTRUCTIONS",
    #     "section style": "Content that Follow the INSTRUCTIONS",
    #     "section style": "Content that Follow the INSTRUCTIONS",
    #     "section style": "Content that Follow the INSTRUCTIONS",
    #     "section style": "Content that Follow the INSTRUCTIONS"
    #         }}
    #     [/Assistant_Example]

    #     [User]
    #     final summary = {all_usr_data}
    #     section list = {structure}
    #     [/User]
    #     """

    #     request = {
    #         "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
    #         "prompt": prompt,
    #         "max_tokens": max_tokens,
    #         "temperature": 0.7,
    #         "top_p": 0.3
    #     }

    #     result = await self.batch_handler.process_single_request(request, 0)
        
    #     if result.success:
    #         response = result
    #         print(f"Section content response: {response}")
    #         return response
    #     else:
    #         print(f"Section content generation error: {result.error}")
    #         return ""


class OpenAISectionGenerator:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler
        self.structure_generator = OpenAISectionStructureGenerator(batch_handler)
        self.content_generator = OpenAISectionContentGenerator(batch_handler)

    async def generate_landing_page(self, requests, max_tokens: int = 200):
        results = []

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
            section_structure_LLM_result = await self.structure_generator.create_section_structure(combined_data, max_tokens)
            print(f"section_structure_LLM_result.data.generations[0][0].text : {section_structure_LLM_result.data.generations[0][0].text}")
            section_structure = section_structure_LLM_result.data.generations[0][0].text.strip()
            # section_structure = await self.structure_generator.create_section_structure(combined_data)
            section_structure_LLM_result.data.generations[0][0].text = self.extract_json(section_structure)
                
            updated_structure = {}
            section_structure = section_structure_LLM_result.data.generations[0][0].text


            updated_structure = {}
            used_values = set()  # 이미 사용된 값을 추적

            for section, value in section_structure.items():
                if section in allowed_values:
                    if value not in allowed_values[section]:
                        # 허용되지 않은 값일 경우 해당 섹션의 허용된 값 중 무작위 선택
                        print()
                        value = random.choice(allowed_values[section])
                    
                    # 이미 사용된 값인지 확인
                    if value not in used_values:
                        # 처음 등장하는 값이면 추가
                        used_values.add(value)
                        updated_structure[section] = value
                    # 이미 사용된 값이면 해당 섹션은 건너뜀 (중복 제거)
                else:
                    print(f"Warning: Undefined section '{section}' encountered.")

                    
            section_structure_LLM_result.data.generations[0][0].text = updated_structure
            
            valid_dict = section_structure_LLM_result.data.generations[0][0].text
            if not isinstance(section_structure_LLM_result.data.generations[0][0].text, dict):
                print(f"===== Retry create_section_structure...count {cnt}=====")
                print(f"===== Because The condition is not fited : {valid_dict}")
                cnt += 1
            else:
                break
        create_section = section_structure_LLM_result.data.generations[0][0].text
        print(f"create_section : {create_section}")
        section_content_data = await self.content_generator.generate_section_contents_individually(
            create_section,
            combined_data,
            max_tokens=300
        )

        print("check", {
            "section_structure": section_structure_LLM_result,
            "section_contents": section_content_data
        })
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