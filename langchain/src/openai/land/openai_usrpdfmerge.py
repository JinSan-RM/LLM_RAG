from langchain.prompts import PromptTemplate
import asyncio
import re
import json

class OpenAIDataMergeClient:
    def __init__(self, usr_msg, pdf_data, batch_handler):
        self.usr_msg = usr_msg
        self.pdf_data = pdf_data
        self.batch_handler = batch_handler

    async def contents_merge(self, max_tokens: int = 1500) -> str:
        max_retries = 3
        for attempt in range(max_retries):
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

                response = await asyncio.wait_for(
                    self.batch_handler.process_single_request({
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "repetition_penalty": 1.2,
                        "frequency_penalty": 1.0,
                        "n": 1,
                        "stream": False,
                        "logprobs": None
                    }, request_id=0),
                    timeout=120
                )
                print("=========== USR_PDF_MERGE ===========")
                print(f"extracted_text_usr_pdf_merge : {response.data.generations[0][0].text}")
                print(f"All_response_of_usr_pdf_merge : {response}")
                print("======================================")

                # 응답 구조 확인 및 텍스트 추출
                if isinstance(response.data, dict) and 'generations' in response.data:
                    generated_text = response.data['generations'][0]['text']
                elif hasattr(response.data, 'generations'):
                    generated_text = response.data.generations[0][0].text
                else:
                    raise ValueError("Unexpected response structure")

                generated_text = self.clean_text(generated_text)
                print(f"response_merge_result: {generated_text}")

                if len(generated_text) <= 50:
                    print(f"Generated text is too short (attempt {attempt + 1}). Retrying...")
                    if attempt == max_retries - 1:
                        return "텍스트 생성 실패: 생성된 텍스트가 너무 짧습니다."
                    continue
                response.data.generations[0][0].text = generated_text
                return response

            except asyncio.TimeoutError:
                print(f"Contents merge request timed out (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    return "텍스트 생성 실패: 시간 초과"
            except Exception as e:
                print(f"Error in contents_merge (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return f"텍스트 생성 실패: {str(e)}"

        return "텍스트 생성 실패: 최대 재시도 횟수 초과"

    # def extract_text(self, result):
    #     """BatchRequestHandler의 결과에서 텍스트를 추출합니다."""
    #     logger.debug(f"result final: {result}")
    #     if result.success and result.data:
    #         if isinstance(result.data, dict) and "choices" in result.data:
    #             text = result.data["choices"][0]["message"]["content"]
    #         elif hasattr(result.data, "generations"):
    #             text = result.data.generations[0][0].text
    #         else:
    #             text = str(result.data)
    #         cleaned_text = self.clean_text(text)
    #         logger.debug(f"cleaned_text: {cleaned_text}")
    #         return cleaned_text
    #     return "텍스트 생성 실패"

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
                "USER_EXAMPLE",
                "[END]",
                "[OUTPUT]"
            ]
        cleaned_text = text
        for header in headers_to_remove:
            cleaned_text = cleaned_text.replace(header, '')
        cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text)
        return cleaned_text.strip()