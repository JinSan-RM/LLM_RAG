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
        
        
# TODO 250501 : 한 번에 끝낼거라서 이 부분은 없어져도 됨. 다만 prompt 참고
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
    
    # # 이부분 코쳐야함
    # def create_section_prompt(self, section_name, combined_data):
    #     # # 입력 언어 감지
    #     # is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in combined_data)
    #     # language_instruction = "한국어로 작성하세요." if is_korean else "Write in English."

    #     sys_prompt = f"""
    #     [System]
    #     You are a professional content generator for sections in the website landing pages.
    #     You know what should be included in the composition of each section.
    #     Your task is to generate concise, unique content based on combined_data.

    #     #### INSTRUCTIONS ####
    #     1. WRITE PLAIN TEXT CONTENT FOR THE 'section_name' SECTION.
    #     2. WRITE BETWEEN 200 AND 300 CHARACTERS IN FOR THE OUTPUT.
    #     3. USE ONLY RELEVANT PARTS OF THE USER'S DATA: 'combined_data'.
    #     4. AVOID REPEATING CONTENT.
    #     5. DO NOT include ANY structure, tags, headers (e.g., ###, [System], [Response]), or metadata. Output ONLY the raw text.
        
    #     #### Output Language ####
    #     **{self.output_language}**
    #     """
        
    #     usr_prompt = f"""
    #     Section_name = {section_name}
    #     all_usr_data = {combined_data}
    #     """
    #     return sys_prompt, usr_prompt
    
    # async def generate_section_contents_individually(self, section_structure, combined_data, max_tokens=300):
    #     # 모든 섹션에 대한 태스크 생성
    #     tasks = []
    #     for section_key, section_name in section_structure.items():
    #         sys_prompt, usr_prompt = self.create_section_prompt(section_name, combined_data)
    #         tasks.append(self.send_request(sys_prompt=sys_prompt, usr_prompt=usr_prompt, max_tokens=max_tokens))
        
    #     # 모든 태스크를 병렬로 실행
    #     results = {}
    #     responses = await asyncio.gather(*tasks)
        
    #     # 결과 처리
    #     for section_key, response in zip(section_structure.keys(), responses):
    #         section_name = section_structure[section_key]
    #         if response.success and hasattr(response, 'data'):
    #             # content = response.data.generations[0][0].text
    #             content = response.data['generations'][0][0]['text']
    #             # 콘텐츠 정제
    #             content = self.clean_content(content)
    #             results[section_name] = content
        
    #     return results

    def clean_content(self, content):
        # 불필요한 태그, 형식, 메타데이터 제거
        content = re.sub(r'\[.*?\]|\#\#\#|\n|\t|Output:|Example|\*\*.*?\*\*|Response:|`|Title:|Description:|Content:', '', content).strip()
        content = re.sub(r'.*?content for.*?from the context\.', '', content, flags=re.IGNORECASE)
        content = ' '.join(content.split())
        return content


class OpenAISectionGenerator:
    def __init__(self, output_language, batch_handler: BatchRequestHandler):
        output_language=output_language
        self.batch_handler = batch_handler
        
        # TODO 250501 : 아래가 content가 될 것
        self.structure_generator = OpenAISectionStructureGenerator(batch_handler)
        # self.content_generator = OpenAISectionContentGenerator(output_language, batch_handler)
        
    async def generate_subpage(self, requests, max_tokens: int = 200):
        results = []
        extra_body = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "sections": {
                            "type": "object",
                            "description": "해당 섹션의 내용입니다."
                        },
                        "minItems": 3,
                        "maxItems": 6
                    }
                },
                "required": ["sections"]
            }
        }
        
        self.structure_generator.set_extra_body(extra_body)
        for req in requests:

            section_data = await self.generate_section(req.sub_context, max_tokens)
            results.append(section_data)
        return results

    # NOTE : merged된 데이터가 들어오면서 기존 2개를 합치던 방식이 1개로 바뀜
    async def generate_section(self, sub_context: str, max_tokens: int = 200):
        combined_data = f"sub_context = {sub_context}"

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


        section_structure_LLM_result.data['generations'][0][0]['text'] = updated_structure

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