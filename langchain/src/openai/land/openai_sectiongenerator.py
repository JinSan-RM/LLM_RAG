import json
import re
import asyncio
from src.utils.batch_handler import BatchRequestHandler


class OpenAISectionStructureGenerator:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler

    async def send_request(self, prompt: str) -> str:
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.1,
                "top_p": 0.1,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=300  # 적절한 타임아웃 값 설정
        )
        return response
    
    async def create_section_structure(self, final_summary_data: str):
        
        prompt = f"""
        [System]
        You are a professional designer who creates website landing page.
        Combine the sections according to the Section style list, Instructions and User input.
        
        #### Section style list ####
        
        - 1 section: ["Hero"]
        - 2 section: ["Feature", "Content"]
        - 3 section: ["CTA", "Feature", "Content", "Gallery", "Comparison", "Logo"]
        - 4 section: ["Gallery", "Comparison", "Statistics", "Timeline", "Countdown", "CTA"]
        - 5 section: ["Testimonial", "Statistics", "Pricing", "FAQ", "Timeline"]
        - 6 section: ["Contact", "FAQ", "Logo", "Team", "Testimonial", "Pricing"]
        
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
        "1 section": "Hero",
        "2 section": "section style",
        "3 section": "section style",
        "4 section": "section style",
        "5 section": "section style",
        "6 section": "section style"
        }}

        [/Assistant_Example]
        
        [User]
        final summary data = {final_summary_data}
        [/User]
        """

        result = await self.send_request(prompt)
             
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

    async def create_section_contents(self, all_usr_data: str, structure: dict):        
        
        prompt = f"""
        [System]
        You are a professional planner who organizes the content of your website landing page.
        Write in the section content by summarizing and distributing the user's final summary data.

        #### INSTRUCTIONS ####
        1. CHECK THE FINAL SUMMARY FOR NECESSARY INFORMATION AND ORGANIZE IT APPROPRIATELY FOR EACH SECTION.
        2. FOR EACH SECTION, PLEASE WRITE ABOUT 200 TO 300 CHARACTERS SO THAT THE CONTENT IS RICH AND CONVEYS THE CONTENT.
        3. PLEASE WRITE WITHOUT TYPOS.
        4. LOOK AT THE INPUT AND **FOLLOW THE INPUT LANGUAGE TO THE OUTPUT**.
        5. ENSURE THAT THE OUTPUT MATCHES THE JSON OUTPUT EXAMPLE BELOW.
        [/System]

        [User_Example]
        final summary = "all_usr_data"
        section list = "section structure"
        [/User_Example]

        [Assistant_Example]
        section_content : {{
        "Hero": "Content that Follow the instructions",
        "section style": "Content that Follow the instructions",
        "section style": "Content that Follow the instructions",
        "section style": "Content that Follow the instructions",
        "section style": "Content that Follow the instructions",
        "section style": "Content that Follow the instructions"
            }}
        [/Assistant_Example]

        [User]
        final summary = {all_usr_data}
        section list = {structure}
        [/User]
        """

        request = {
            "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
            "prompt": prompt,
            "temperature": 0.7,
            "top_p": 0.3
        }

        result = await self.batch_handler.process_single_request(request, 0)
        
        
        if result.success:
            response = result
            print(f"Section content response: {response}")
            return response
        else:
            print(f"Section content generation error: {result.error}")
            return ""


class OpenAISectionGenerator:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler
        self.structure_generator = OpenAISectionStructureGenerator(batch_handler)
        self.content_generator = OpenAISectionContentGenerator(batch_handler)

    async def generate_landing_page(self, requests):
        results = []

        for req in requests:

            section_data = await self.generate_section(req.all_usr_data)
            results.append(section_data)
        return results

    # NOTE : merged된 데이터가 들어오면서 기존 2개를 합치던 방식이 1개로 바뀜
    async def generate_section(self, all_usr_data: str):
        combined_data = f"PDF summary data = {all_usr_data}"
        cnt = 0
        while cnt < 3:
            section_structure_LLM_result = await self.structure_generator.create_section_structure(combined_data)
            section_structure = section_structure_LLM_result.data.generations[0][0].text.strip()
            # section_structure = await self.structure_generator.create_section_structure(combined_data)
            section_structure_LLM_result.data.generations[0][0].text = self.extract_json(section_structure)
            if not isinstance(section_structure_LLM_result.data.generations[0][0].text, dict):
                cnt += 1
            else:
                break
        
        
        # print("HEEEEERE : ", section_structure.data.generations[0][0].text , type(section_structure.data.generations[0][0].text))      
        # section_structure_LLM_result.data.generations[0][0].text = asdf
        cnt = 0
        while cnt < 3:
            
            section_contents = await self.content_generator.create_section_contents(
                combined_data,
                section_structure_LLM_result.data.generations[0][0].text
                )
            contents = section_contents.data.generations[0][0].text.strip()
            section_contents.data.generations[0][0].text = self.extract_json(contents)
            if not isinstance(section_contents.data.generations[0][0].text, dict):
                cnt += 1
            else:
                break
        
        return {
            "section_structure": section_structure_LLM_result,
            "section_contents": section_contents
        }
        
    def extract_json(self, text):
        # 가장 바깥쪽의 중괄호 쌍을 찾습니다.
        print("Herere!!!!! : ", text)
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