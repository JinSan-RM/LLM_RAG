import json
from typing import List, Dict
from src.utils.batch_handler import BatchRequestHandler

class OpenAISectionGenerator:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler

    async def generate_landing_page(self, requests):
        results = []
        for req in requests:
            section_data = await self.generate_section(req.usr_msg, req.pdf_data1)
            results.append(section_data)
        return results

    async def generate_section(self, usr_msg: str, pdf_data1: str):
        combined_data = f"User Message: {usr_msg}\nPDF Data: {pdf_data1}"
        
        section_structure = await self.create_section_structure(combined_data)
        section_contents = await self.create_section_contents(combined_data, section_structure)
        
        return {
            "section_structure": section_structure,
            "section_contents": section_contents
        }

    async def create_section_structure(self, final_summary_data: str):
        prompt = f"""
			System:
			You are a professional designer who creates website landing pages.
			Select the section list according to the Instructions and User input values below.
			
			#### Instructions ####
			
			Step 1. “Hero” is definitely included.
			Step 2. The number of sections **must be between 4 and 6**.
			Step 3. Depending on the order of the sections, freely select the appropriate sections one by one from the section list below.
			Step 4. Be careful not to duplicate section names.
			Step 5. Be careful about typos.
			Seep 6. ensure that the output mathces the JSON output example below.
			
			#### Section lists ####
			- 1st section: [Hero]
			- 2nd section: [Feature, Content]
			- 3rd section: [CTA, Feature, Content, Gallery, Comparison, Logo]
			- 4th section: [Gallery, Comparison, Statistics, Timeline, Countdown, CTA]
			- 5th section: [Testimonial, Statistics, Pricing, FAQ, Timeline]
			- 6th section: [Contact, FAQ, Logo, Team, Testimonial, Pricing]
			
			#### Example ####
			    "menu_structure": {{
			        "1": "section",
			        "2": "section",
			        "3": "section",
			        "4": "section",
			        "5": "section",
			        "6": "section"
			    }}
			
			User:
			{final_summary_data}
        """
        
        request = {
            "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
            "prompt": prompt,
            "temperature": 0.7,
            "top_p": 0.8
        }
        
        result = await self.batch_handler.process_single_request(request, 0)
        if result.success:
            response_text = result.data.generations[0][0].text.strip()
            print(f" proposal response_text : {response_text}")
            return response_text  # 생성된 텍스트를 직접 반환
        else:
            print(f"섹션 구조 섹션 생성 오류: {result.error}")
            return ""

    async def create_section_contents(self, data: str, structure: dict):
        prompt = f"""
        Generate content for each section based on the following structure:
        {json.dumps(structure, indent=2)}

        Input data:
        {data}

        Provide brief content for each section in JSON format.
        """
        
        request = {
            "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
            "prompt": prompt,
            "temperature": 0.7,
            "top_p": 0.8
        }
        
        result = await self.batch_handler.process_single_request(request, 0)
        if result.success:
            response_text = result.data.generations[0][0].text.strip()
            print(f" proposal response_text : {response_text}")
            return response_text  # 생성된 텍스트를 직접 반환
        else:
            print(f"섹션 구조 섹션 생성 오류: {result.error}")
            return ""