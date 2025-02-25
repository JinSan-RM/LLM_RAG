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
                # 언어 감지: usr_msg의 언어를 기준으로 설정
                is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in self.usr_msg)
                output_language = "Korean" if is_korean else "English"

                prompt = f"""
                You are an expert in writing business plans. Write a single business plan by combining the user summary and PDF summary data provided below. Follow these instructions precisely:

                #### INSTRUCTIONS ####
                1. Prioritize the user summary over the PDF summary data when combining the information.
                2. Detect the language of the user summary and write the output entirely in that language (e.g., if the user summary is in Korean, output in Korean; if in English, output in English). The output language must be {output_language}.
                3. Ensure the business plan includes these seven elements. If information is missing, use creativity to fill gaps:
                    1) BUSINESS ITEM: Specific product or service details
                    2) SLOGAN OR CATCH PHRASE: A sentence expressing the company's main vision or ideology
                    3) TARGET CUSTOMERS: Characteristics and needs of the major customer base
                    4) CORE VALUE PROPOSITION: Unique value provided to customers
                    5) PRODUCT AND SERVICE FEATURES: Main functions and advantages
                    6) BUSINESS MODEL: Processes that generate profits by providing differentiated value
                    7) PROMOTION AND MARKETING STRATEGY: How to introduce products or services to customers
                4. Output ONLY the final business plan text. Do not include any tags (e.g., [SYSTEM], [USER]), headers, metadata, JSON formatting (e.g., {{Output: ...}}), or tokens (e.g., <|eot_id|>).
                5. Write between 500 and 1000 characters in {output_language}.
                6. Blend the user summary and PDF summary data evenly in the narrative.
                7. Ignore any structural data (e.g., generations=...) in the PDF summary and extract only meaningful content.

                user summary = {self.usr_msg}
                pdf summary data = {self.pdf_data}
                """

                response = await asyncio.wait_for(
                    self.batch_handler.process_single_request({
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "repetition_penalty": 1.2,
                        "frequency_penalty": 1.0,
                        "n": 1,
                        "stream": False,
                        "logprobs": None
                    }, request_id=0),
                    timeout=120
                )

                # 응답에서 텍스트 추출
                if isinstance(response.data, dict) and 'generations' in response.data:
                    generated_text = response.data['generations'][0]['text']
                elif hasattr(response.data, 'generations'):
                    generated_text = response.data.generations[0][0].text
                else:
                    raise ValueError("Unexpected response structure")

                # 텍스트 정리
                cleaned_text = self.clean_text(generated_text)

                # 길이 확인
                if len(cleaned_text) < 50:
                    print(f"Generated text too short (attempt {attempt + 1}). Retrying...")
                    if attempt == max_retries - 1:
                        return "텍스트 생성 실패: 생성된 텍스트가 너무 짧습니다." if is_korean else "Text generation failed: Generated text too short."
                    continue

                return cleaned_text

            except asyncio.TimeoutError:
                print(f"Contents merge timed out (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    return "텍스트 생성 실패: 시간 초과" if is_korean else "Text generation failed: Timeout"
            except Exception as e:
                print(f"Error in contents_merge (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return f"텍스트 생성 실패: {str(e)}" if is_korean else f"Text generation failed: {str(e)}"

        return "텍스트 생성 실패: 최대 재시도 횟수 초과" if is_korean else "Text generation failed: Max retries exceeded"
    
    def clean_text(self, text):
        # 제거할 패턴 (태그, 메타데이터만 제거, 내용은 유지)
        patterns = [
            r'\[(SYSTEM|USER|END|OUTPUT)\]',  # [SYSTEM], [USER] 등
            r'<\|start_header_id\|>.*?<\|end_header_id\|>',  # <|start_header_id|>... <|end_header_id|>
            r'<\|eot_id\|>',  # <|eot_id|>
            r'\{Output\s*:\s*"(.*?)"\}',  # {Output: "..."} -> "..."만 남김
            r'generations=\[\[.*?(text=)?["\']?(.*?)(["\']|\}\]).*?\]\]',  # generations=... -> 내용만 추출
            r'pdf\s*text\s*=\s*".*?"',  # pdf text="..."
            r'user\s*summary\s*=\s*',  # user summary =
            r'pdf\s*summary\s*data\s*=\s*',  # pdf summary data =
            r'\b(ASSISTANT(_EXAMPLE)?|USER(_EXAMPLE)?|SYSTEM)\b',  # ASSISTANT, USER 등
            r'\n\s*\n',  # 불필요한 줄바꿈
        ]
        
        cleaned_text = text
        for pattern in patterns:
            if pattern == r'\{Output\s*:\s*"(.*?)"\}' or pattern == r'generations=\[\[.*?(text=)?["\']?(.*?)(["\']|\}\]).*?\]\]':
                # 캡처된 내용을 유지
                cleaned_text = re.sub(pattern, r'\2' if 'generations' in pattern else r'\1', cleaned_text, flags=re.DOTALL)
            else:
                # 태그만 제거
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)
        
        return cleaned_text.strip()
