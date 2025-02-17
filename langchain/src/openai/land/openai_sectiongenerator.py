import json
import asyncio
from typing import List, Dict
from src.utils.batch_handler import BatchRequestHandler

class OpenAISectionGenerator:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler

    async def generate_landing_page(self, requests):
        print("generate_landing_page 시작")
        results = []
        for i, req in enumerate(requests):
            print(f"요청 {i+1} 처리 중")
            section_data = await self.generate_section(req.usr_msg, req.pdf_data1)
            results.append(section_data)
        print("generate_landing_page 완료")
        return results

    async def generate_section(self, usr_msg: str, pdf_data1: str):
        print("generate_section 시작")
        combined_data = f"User Message: {usr_msg}\nPDF Data: {pdf_data1}"
        print(f"Combined data: {combined_data[:100]}...")  # 데이터의 일부만 출력
        
        print("섹션 구조 생성 중...")
        section_structure = await self.create_section_structure(combined_data)
        print(f"생성된 섹션 구조: {section_structure}")
        
        print("섹션 내용 생성 중...")
        section_contents = await self.create_section_contents(combined_data, section_structure)
        print(f"생성된 섹션 내용: {section_contents}")
        
        print("generate_section 완료")
        return {
            "section_structure": section_structure,
            "section_contents": section_contents
        }
    async def create_section_structure(self, data: str):
        print("create_section_structure 시작")
        prompt = f"""
        Generate a website landing page section structure following these rules:
        1. Hero section is mandatory.
        2. Create 4-6 sections in total.
        3. Possible sections: Hero, Feature, Content, CTA, Gallery, Comparison, Logo, Statistics, Testimonial, Pricing, FAQ, Contact, Team
        4. Respond in JSON format.

        Input data:
        {data[:100]}...  # 데이터의 일부만 프롬프트에 포함
        """
        print(f"생성된 프롬프트: {prompt[:100]}...")  # 프롬프트의 일부만 출력
        
        try:
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
                timeout=60
            )
            print("create_section_structure 완료")
            return response
        except asyncio.TimeoutError:
            print("create_section_structure 시간 초과")
            return None
        except Exception as e:
            print(f"create_section_structure 오류 발생: {str(e)}")
            return None

    async def create_section_contents(self, data: str, structure: dict):
        print("create_section_contents 시작")
        prompt = f"""
        Generate content for each section based on the following structure:
        {{}}

        Input data:
        {data[:100]}...  # 데이터의 일부만 프롬프트에 포함

        Provide brief content for each section in JSON format.
        """
        print(f"생성된 프롬프트: {prompt[:100]}...")  # 프롬프트의 일부만 출력
        
        try:
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
                timeout=60
            )
            print("create_section_contents 완료")
            return response
        except asyncio.TimeoutError:
            print("create_section_contents 시간 초과")
            return None
        except Exception as e:
            print(f"create_section_contents 오류 발생: {str(e)}")
            return None