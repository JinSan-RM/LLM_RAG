import asyncio
from typing import List
import json
from vllm import SamplingParams
from langchain.prompts import PromptTemplate

class OpenAISummaryClient:
    def __init__(self, pdf_data: str, batch_handler):
        self.pdf_data = pdf_data
        self.batch_handler = batch_handler
        

    async def summarize_text(self, text: str, desired_summary_length: int) -> str:
        try:
            prompt = f"Summarize the following text in about {desired_summary_length} characters:\n\n{text}"
            
            request = {
                "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
                "prompt": prompt,
                "max_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.8
            }

            result = await asyncio.wait_for(
                self.batch_handler.process_single_request(request, 0),
                timeout=120
            )
            if result.success:
                return result.data.generations[0][0].text.strip()
            else:
                print(f"Error in summarizing text: {result.error}")
                return ""

        except asyncio.TimeoutError:
            print("Summarization request timed out")
            return ""

    async def process_pdf(self):
        try:
            summary = await self.summarize_text(self.pdf_data, 3000)
            if not summary:
                print("Failed to generate summary")
                return ""
            
            proposal = await self.generate_proposal(summary)
            if not proposal:
                print("Failed to generate proposal")
                return ""
            
            return proposal
        except Exception as e:
            print(f"Error in processing PDF: {str(e)}")
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
            
            request = {
                "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.1,
                "top_p": 0.8
            }
            
            result = await asyncio.wait_for(
                self.batch_handler.process_single_request(request, 0),
                timeout=120
            )
            if result.success:
                response_text = result.data.generations[0][0].text.strip()
                try:
                    json_response = json.loads(response_text)
                    return json.dumps(json_response, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    print("Failed to parse JSON response")
                    return response_text
            else:
                print(f"Error in generating proposal: {result.error}")
                return ""
        except asyncio.TimeoutError:
            print("Summary proposal request timed out")
            return ""
