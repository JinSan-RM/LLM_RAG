from langchain.prompts import PromptTemplate
import asyncio
import re
import json

class OpenAIDataMergeClient:
    def __init__(self, usr_msg, pdf_data, batch_handler):
        # JSON 문자열에서 텍스트 추출
        if isinstance(usr_msg.data.generations[0][0].text, str):
            try:
                data = json.loads(usr_msg.data.generations[0][0].text.replace("'", '"'))
                self.usr_msg = data.get("Output", usr_msg.data.generations[0][0].text)
            except json.JSONDecodeError:
                self.usr_msg = usr_msg.data.generations[0][0].text
        else:
            self.usr_msg = str(usr_msg.data.generations[0][0].text)

        if isinstance(pdf_data.data.generations[0][0].text, str):
            try:
                data = json.loads(pdf_data.data.generations[0][0].text.replace("'", '"'))
                self.pdf_data = data.get("Output", pdf_data.data.generations[0][0].text)
            except json.JSONDecodeError:
                self.pdf_data = pdf_data.data.generations[0][0].text
        else:
            self.pdf_data = str(pdf_data.data.generations[0][0].text)

        self.batch_handler = batch_handler

    async def contents_merge(self) -> str:
        try:
            prompt = f"""
            [SYSTEM]
            You are an expert in writing business plans. Write a single business plan by combining the user summary and PDF summary data provided below. Follow these instructions precisely:

            #### INSTRUCTIONS ####
            1. Prioritize the user summary over the PDF summary data when combining the information.
            2. Detect the language of the user summary and PDF summary data, and write the output entirely in that language. For example, if the input is in Korean, the output must be in Korean; if in English, the output must be in English.
            3. Ensure the business plan includes the following seven elements. If information is missing, use creativity to fill in the gaps based on the user summary and PDF summary data:
                1) BUSINESS ITEM: Specific product or service details
                2) SLOGAN OR CATCH PHRASE: A sentence expressing the company's main vision or ideology
                3) TARGET CUSTOMERS: Characteristics and needs of the major customer base
                4) CORE VALUE PROPOSITION: Unique value provided to customers
                5) PRODUCT AND SERVICE FEATURES: Main functions and advantages
                6) BUSINESS MODEL: Processes that generate profits by providing differentiated value to customers
                7) PROMOTION AND MARKETING STRATEGY: How to introduce products or services to customers
            4. Output only the final business plan text, without any additional tags, headers, or metadata. Do not include <|eot_id|> or similar tokens in the output.
            5. Write between 500 and 1000 characters to ensure rich and detailed content.
            6. Blend the user summary and PDF summary data evenly in the narrative.
            7. Do not include any system prompts, tags like [SYSTEM], or JSON formatting in the output.

            [USER]
            user summary = {self.usr_msg}
            pdf summary data = {self.pdf_data}
            """
            print(f"usr_msg: {self.usr_msg}")
            print(f"pdf_data: {self.pdf_data}")
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                    "prompt": prompt,
                    "max_tokens": 2000,
                    "temperature": 0.7,  # 약간 높여 창의성 확보
                    "top_p": 0.9,       # 덜 보수적으로 조정
                    "repetition_penalty": 1.2,  # 반복 억제 완화
                    "frequency_penalty": 1.0,   # 빈도 억제 완화
                    "n": 1,
                    "stream": False,
                    "logprobs": None
                }, request_id=0),
                timeout=60
            )
            response.data.generations[0][0].text = self.extract_text(response)
            print(f"merge_result: {response.data.generations[0][0].text}")
            return response
        except asyncio.TimeoutError:
            print("Contents merge request timed out")
            return ""
        except Exception as e:
            print(f"Error in contents_merge: {e}")
            return ""

    def extract_text(self, result):
        print(f"result final: {result}")
        if result.success and result.data.generations:
            text = result.data.generations[0][0].text.strip()
            # 태그 및 불필요한 토큰 제거
            cleaned_text = self.clean_text(text)
            print(f"cleaned_text: {cleaned_text}")
            return cleaned_text
        return "텍스트 생성 실패"

    def clean_text(self, text):
        headers_to_remove = [
                "<|start_header_id|>system<|end_header_id|>",
                "<|start_header_id|>SYSTEM<|end_header_id|>",
                "<|start_header_id|>", "<|end_header_id|>",
                "<|start_header_id|>user<|end_header_id|>",
                "<|start_header_id|>assistant<|end_header_id|>",
                "<|eot_id|><|start_header_id|>ASSISTANT_EXAMPLE<|end_header_id|>",
                "<|eot_id|><|start_header_id|>USER_EXAMPLE<|end_header_id|>",
                "<|eot_id|><|start_header_id|>USER<|end_header_id|>",
                "<|eot_id|>",
                "<|eot_id|><|start_header_id|>ASSISTANT<|end_header_id|>",
                "ASSISTANT",
                "USER",
                "SYSTEM",
                "<|end_header_id|>",
                "<|start_header_id|>"
                "ASSISTANT_EXAMPLE",
                "USER_EXAMPLE"
            ]
        cleaned_text = text
        for header in headers_to_remove:
            cleaned_text = cleaned_text.replace(header, '')
        cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text)
        return cleaned_text.strip()