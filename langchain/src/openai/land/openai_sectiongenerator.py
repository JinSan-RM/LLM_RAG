import json
import asyncio
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

    async def create_section_structure(self, data: str):
        prompt = f"""
        Generate a website landing page section structure following these rules:
        1. Hero section is mandatory.
        2. Create 4-6 sections in total.
        3. Possible sections: Hero, Feature, Content, CTA, Gallery, Comparison, Logo, Statistics, Testimonial, Pricing, FAQ, Contact, Team
        4. Respond in JSON format.

        Input data:
        {data}
        """
        
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 1.0,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=60  # 적절한 타임아웃 값 설정
        )
        return response

    async def create_section_contents(self, data: str, structure: dict):
        prompt = f"""
        Generate content for each section based on the following structure:
        {json.dumps(structure, indent=2)}

        Input data:
        {data}

        Provide brief content for each section in JSON format.
        """
        
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 1.0,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=60  # 적절한 타임아웃 값 설정
        )
        return response