from langchain.prompts import PromptTemplate
import asyncio
import re
import json

class OpenAIDataMergeClient:
    def __init__(self, usr_msg, pdf_data, batch_handler):
        self.usr_msg = usr_msg
        self.pdf_data = pdf_data
        self.batch_handler = batch_handler

    async def contents_merge(self, max_tokens: int = 1500) -> dict:
        max_retries = 3
        
        temp_usr = self.usr_msg
        temp_pdf = self.pdf_data
        if len(temp_usr) + len(temp_pdf) > 2000:
            temp_usr = temp_usr[:1000]
            temp_pdf = temp_pdf[:1000]
        
        for attempt in range(max_retries):
            try:
                
                # NOTE : 이 부분은 나중에는 연산을 넣어서 판단하면 될듯
                # 사용자 입력 언어 확인 (한글 여부 판단)
                is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in self.usr_msg)
                output_language = "Korean" if is_korean else "English"

                
                # 6. Write between 500 and 1000 characters in {output_language} for the output.
                # 프롬프트 생성
                prompt = f"""
                [SYSTEM]
                You are an expert in writing business plans. 
                Write a single business plan by combining the user summary and PDF summary data provided below. 
                Follow these instructions precisely:

                #### INSTRUCTIONS ####
                1. Prioritize the user summary over the PDF summary data when combining the information.
                2. Detect the language of the user summary and write the output entirely in that language. The output language must be {output_language}.
                3. Ensure the business plan includes these seven elements. If information is missing, use creativity to fill gaps:
                    1) BUSINESS ITEM: Specific product or service details
                    2) SLOGAN OR CATCH PHRASE: A sentence expressing the company's main vision or ideology
                    3) TARGET CUSTOMERS: Characteristics and needs of the major customer base
                    4) CORE VALUE PROPOSITION: Unique value provided to customers
                    5) PRODUCT AND SERVICE FEATURES: Main functions and advantages
                    6) BUSINESS MODEL: Processes that generate profits by providing differentiated value
                    7) PROMOTION AND MARKETING STRATEGY: How to introduce products or services to customers
                4. Output the final business plan text as a plain, continuous string without any JSON structure, tags, labels, or metadata.
                5. Integrate both the user summary and PDF summary data into a single, cohesive text without separating them or using labels like "Output:" or "pdf text =".
                6. Blend the user summary and PDF summary data evenly in the narrative, ensuring a smooth and logical flow of information.
                7. 출력은 반드시 **한국어**로 해.
                
                [USER]
                user summary = {temp_usr}
                pdf summary data = {temp_pdf}
                """

                # API 호출 및 응답 처리
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

                # 응답에서 생성된 텍스트 추출
                if isinstance(response.data, dict) and 'generations' in response.data:
                    generated_text = response.data['generations'][0]['text']
                elif hasattr(response.data, 'generations'):
                    generated_text = response.data.generations[0][0].text
                else:
                    raise ValueError("Unexpected response structure")

                # 텍스트 정리 (구조 및 태그 제거)
                cleaned_text = self.clean_text(generated_text)

                # 텍스트 길이 확인 및 재시도 처리
                if len(cleaned_text) < 50:
                    if attempt == max_retries - 1:
                        return {"error": "텍스트 생성 실패: 생성된 텍스트가 너무 짧습니다."} if is_korean else {"error": "Text generation failed: Generated text too short."}
                    continue
                response.data.generations[0][0].text = cleaned_text
                return response

            except asyncio.TimeoutError:
                print(f"Contents merge timed out (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    return {"error": "텍스트 생성 실패: 시간 초과"} if is_korean else {"error": "Text generation failed: Timeout"}
            except Exception as e:
                print(f"Error in contents_merge (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return {"error": f"텍스트 생성 실패: {str(e)}"} if is_korean else {"error": f"Text generation failed: {str(e)}"}

        return {"error": "텍스트 생성 실패: 최대 재시도 횟수 초과"} if is_korean else {"error": "Text generation failed: Max retries exceeded"}
    
    def clean_text(self, text):
        text = re.sub(r'\\"', '', text)
    
        # 2. 중복된 큰따옴표 ("") 제거
        text = re.sub(r'""', '', text)
        patterns = [
            r'\[(SYSTEM|USER|END|OUTPUT)\]',
            r'<\|start_header_id\|>.*?<\|end_header_id\|>',
            r'<\|eot_id\|>',
            r'\{Output\s*:\s*.*?\}',  # 수정된 패턴
            r'\{output\s*:\s*.*?\}',  # 수정된 패턴
            r'\{[^}]*\}',  # 모든 중괄호 내용 제거
            r'pdf\s*text\s*=\s*',
            r'user\s*summary\s*=\s*',
            r'pdf\s*summary\s*data\s*=\s*',
            r'\b(ASSISTANT(_EXAMPLE)?|USER(_EXAMPLE)?|SYSTEM)\b',
            r'\n\s*\n',
            r'\*=\*',
            r'\*:\*'
            ":",
            "{",
            "}",
            "\\n",
            '\\"',
            "<|start_header_id|>system<|end_header_id|>",
            "<|start_header_id|>SYSTEM<|end_header_id|>",
            "<|start_header_id|>", "<|end_header_id|>",
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|eot_id|>", "[SYSTEM]", "[USER]", "ASSISTANT",
            "Output"
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 모든 따옴표 제거 (작은따옴표, 큰따옴표)
        text = re.sub(r'[\'"]', '', text)
        
        # 모든 대괄호와 그 내용 제거
        text = re.sub(r'\[.*?\]', '', text)
        
        # 모든 꺾쇠괄호와 그 내용 제거
        text = re.sub(r'<.*?>', '', text)
        
        # 특정 키워드와 그 뒤의 콜론 제거
        text = re.sub(r'\b(Output|output|pdf text|user summary|pdf summary data)\s*:', '', text)
        
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 줄바꿈 문자를 실제 줄바꿈으로 변경
        text = text.replace('\\n', '\n')
        
        return text.strip()


