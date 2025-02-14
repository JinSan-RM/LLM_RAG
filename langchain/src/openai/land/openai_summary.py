import asyncio
import json
import re
from typing import List
from vllm import SamplingParams

class OpenAISummaryClient:
    def __init__(self, pdf_data: str, batch_handler):
        self.pdf_data = pdf_data
        self.batch_handler = batch_handler

    @staticmethod
    def extract_json(text):
        try:
            # 불완전한 JSON 문자열 완성 시도
            text = text.strip()
            if text.endswith(','):
                text = text[:-1]
            if not text.endswith('}'):
                text += '}'
            return json.loads(text)
        except json.JSONDecodeError:
            print("JSON 파싱 실패")
            return None

    async def summarize_text(self, text: str, desired_summary_length: int) -> str:
        try:
            prompt = f"Summarize the following text in about {desired_summary_length} characters:\n\n{text}"
            sampling_params = SamplingParams(max_tokens=2000)
            request = {
                "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
                "sampling_params": sampling_params,
                "prompt": prompt,
                "max_tokens": 2000,
                "temperature": 0.1,
                "top_p": 0.8
            }
            result = await asyncio.wait_for(
            self.batch_handler.process_single_request(request, 0),
            timeout=120
            )
            print(f"summary LLMResult 구조: {type(result.data)}")
            print(f"summary LLMResult 내용: {result.data}")
            if result.success:
                # if hasattr(result.data, 'generations') and result.data.generations:
                response_text = result.data.generations[0][0].text.strip()
                print(f" summary response_text : {response_text}")
                response_text = str(response_text)
                return response_text  # 생성된 텍스트를 직접 반환
            else:
                print(f"summary 요약 생성 오류: {result.error}")
                return ""
        except asyncio.TimeoutError:
            print("요약 요청 시간 초과")
            return ""
        except Exception as e:
            print(f"요약 중 예상치 못한 오류: {str(e)}")
            return ""

    async def generate_proposal(self, summary: str):
        try:
            prompt = f"""
            Create a website proposal based on the following summary:
            {summary}

            The proposal should include:
            - Website name
            - Keywords (7 items)
            - Purpose (200 characters)
            - Target audience (3 items)
            - Core values (3 items)
            - Main service
            - Key achievements (5 items)

            Provide the output in JSON format.
            """
            sampling_params = SamplingParams(max_tokens=4000)
            request = {
                "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
                "sampling_params": sampling_params,
                "prompt": prompt,
                "max_tokens": 2000,
                "temperature": 0.1,
                "top_p": 0.8
            }
            result = await asyncio.wait_for(
                self.batch_handler.process_single_request(request, 0),
                timeout=120
            )
            print(f"proposal LLMResult 구조: {type(result.data)}")
            print(f"proposal LLMResult 내용: {result.data}")
            if result.success:
                # if hasattr(result.data, 'generations') and result.data.generations:
                response_text = result.data.generations[0][0].text.strip()
                print(f" proposal response_text : {response_text}")
                return response_text  # 생성된 텍스트를 직접 반환
            else:
                print(f"제안서 생성 오류: {result.error}")
                return ""
        except Exception as e:
            print(f"제안서 생성 중 예상치 못한 오류: {str(e)}")
            return ""
        
    async def process_pdf(self):
        try:
            summary = await self.summarize_text(self.pdf_data, 2000)
            if not summary:
                print("요약 생성 실패")
                return ""
            
            proposal = await self.generate_proposal(summary)
            if not proposal:
                print("제안서 생성 실패")
                return ""
            
            return proposal  # 이 값을 직접 사용할 수 있습니다
        except Exception as e:
            print(f"PDF 처리 중 오류: {str(e)}")
            return ""