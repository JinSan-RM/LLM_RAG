import asyncio
from langchain.prompts import PromptTemplate

class OpenAIUsrMsgClient:
    def __init__(self, usr_msg, batch_handler):
        self.batch_handler = batch_handler
        self.usr_msg = str(usr_msg) if usr_msg else "입력 내용 없음"

    async def usr_msg_proposal(self, max_tokens: int = 500) -> dict:
        try:
            # 프롬프트 템플릿 정의 (간결하게)
            sys_prompt = """
            당신은 비즈니스 계획 전문가입니다. 주어진 사용자 입력을 기반으로 랜딩페이지용 콘텐츠를 한국어로 500~700자 이내로 작성하세요. 
            입력에서 핵심 정보(제품/서비스, 고객, 가치 등)를 추출하고, 부족하면 적절히 보완하세요. 
            포함할 요소: 사업 아이템, 슬로건, 타겟 고객, 핵심 가치, 기능, 비즈니스 모델, 마케팅 전략. 
            자연스럽고 간결한 문장으로, 태그나 메타데이터 없이 출력하세요.
            """
            usr_prompt = self.usr_msg

            # API 호출
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                    "sys_prompt": sys_prompt,
                    "usr_prompt": usr_prompt,
                    "max_tokens": max_tokens,  # 500~700자에 맞게 조정
                    "temperature": 0.5,  # 자연스러움과 정확성 균형
                    "top_p": 0.8,
                    "repetition_penalty": 1.1,  # 반복 감소
                    "n": 1,
                    "stream": False
                }, request_id=0),
                timeout=30  # 타임아웃 줄여 효율성 증가
            )

            # 응답 검증 및 처리
            if response is None or not hasattr(response, 'data') or 'generations' not in response.data:
                print("API 응답 오류: 유효한 데이터 없음")
                return {"error": "API 응답 오류"}

            text = response.data['generations'][0][0]['text'].replace("\n", " ").strip()
            response.data['generations'][0][0]['text'] = text
            print(f"생성된 콘텐츠 길이: {len(text)}자")
            return response.data  # 딕셔너리 반환
        except asyncio.TimeoutError:
            print("사용자 메시지 처리 타임아웃")
            return {"error": "요청 타임아웃"}
        except Exception as e:
            print(f"usr_msg_proposal 오류: {str(e)}")
            return {"error": str(e)}



# from langchain.prompts import PromptTemplate
# import asyncio
# import re
# import json

# class OpenAIUsrMsgClient:
#     def __init__(self, usr_msg, batch_handler):
#         self.batch_handler = batch_handler
#         self.usr_msg = str(usr_msg)
            

#     async def usr_msg_proposal(self, max_tokens: int = 1000):
#         try:
            
#             # NOTE : 이 부분은 나중에는 연산을 넣어서 판단하면 될듯
#             is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in self.usr_msg)
#             output_language = "Korean" if is_korean else "English"            
#             # STEP 1. Read the user input carefully. The default language is {output_language}
#             # prompt = f"""
#             # [SYSTEM]
#             # You are a professional in business plan writing. 
#             # You are provided with user input in the form of a sentence or paragraph. 
#             # Your task is to write a narrative paragraph to assist in creating a business plan based on this input. 
#             # Follow these instructions precisely:

#             # #### INSTRUCTIONS ####
            
#             # STEP 1. Identify and include key information from the user input, such as below. If the usr_input is not enough, you can fill it yourself.
#             #     1) BUSINESS ITEM: Specific product or service details
#             #     2) SLOGAN OR CATCH PHRASE: A sentence expressing the company's main vision or ideology
#             #     3) TARGET CUSTOMERS: Characteristics and needs of the major customer base
#             #     4) CORE VALUE PROPOSITION: Unique value provided to customers
#             #     5) PRODUCT AND SERVICE FEATURES: Main functions and advantages
#             #     6) BUSINESS MODEL: Processes that generate profits by providing differentiated value
#             #     7) PROMOTION AND MARKETING STRATEGY: How to introduce products or services to customers            
#             # STEP 2. Develop the business plan narrative step-by-step using only the keywords and details from the user input. Do not expand the scope beyond the provided content.
#             # STEP 3. Write a paragraph of 1000 to 1500 characters to ensure the content is detailed and informative.
#             # STEP 4. Avoid repeating the same content to meet the character limit. Use varied expressions and vocabulary to enrich the narrative.
#             # STEP 5. Do not include repeated User-Assistant interactions or unnecessary filler. End the output naturally after completing the narrative.
#             # STEP 6. Ensure the text is free of typos and grammatical errors.
#             # STEP 7. Output only the final business plan narrative text. Do not include tags (e.g., [SYSTEM], <|eot_id|>), JSON formatting (e.g., {{Output: "..."}}), or any metadata in the output.
#             # STEP 8. 출력은 반드시 **한국어**로 해.
            
#             # [USER]
#             # user_input = {self.usr_msg}
#             # """
#             sys_prompt = f"""
#             You are a professional in business plan writing. 
#             You are provided with user input in the form of a sentence or paragraph. 
#             Your task is to write a narrative paragraph to assist in creating a business plan based on this input. 
#             Follow these instructions precisely:

#             #### INSTRUCTIONS ####
            
#             STEP 1. Identify and include key information from the user input, such as below. If the usr_input is not enough, you can fill it yourself.
#                 1) BUSINESS ITEM: Specific product or service details
#                 2) SLOGAN OR CATCH PHRASE: A sentence expressing the company's main vision or ideology
#                 3) TARGET CUSTOMERS: Characteristics and needs of the major customer base
#                 4) CORE VALUE PROPOSITION: Unique value provided to customers
#                 5) PRODUCT AND SERVICE FEATURES: Main functions and advantages
#                 6) BUSINESS MODEL: Processes that generate profits by providing differentiated value
#                 7) PROMOTION AND MARKETING STRATEGY: How to introduce products or services to customers            
#             STEP 2. Develop the business plan narrative step-by-step using only the keywords and details from the user input. Do not expand the scope beyond the provided content.
#             STEP 3. Write a paragraph of 1000 to 1500 characters to ensure the content is detailed and informative.
#             STEP 4. Avoid repeating the same content to meet the character limit. Use varied expressions and vocabulary to enrich the narrative.
#             STEP 5. Do not include repeated User-Assistant interactions or unnecessary filler. End the output naturally after completing the narrative.
#             STEP 6. Ensure the text is free of typos and grammatical errors.
#             STEP 7. Output only the final business plan narrative text. Do not include tags (e.g., [SYSTEM], <|eot_id|>), JSON formatting (e.g., {{Output: "..."}}), or any metadata in the output.
#             STEP 8. 출력은 반드시 **한국어**로 해.
#             """
#             usr_prompt = f'''
#             {self.usr_msg}
#             '''         
#             response = await asyncio.wait_for(
#                 self.batch_handler.process_single_request({
#                     "sys_prompt": sys_prompt,
#                     "usr_prompt": usr_prompt,
#                     "max_tokens": max_tokens,  # 1000~1500자 출력 위해 충분히 설정
#                     "temperature": 0.7,
#                     "top_p": 0.9,  # 더 자연스러운 출력 위해 조정
#                     "repetition_penalty": 1.2,
#                     "frequency_penalty": 1.0,
#                     "n": 1,
#                     "stream": False,
#                     "logprobs": None
#                 }, request_id=0),
#                 timeout=100
#             )
#             re_text = response.data['generations'][0][0]['text']
#             re_text = re_text.replace("\n", " ")
#             response.data['generations'][0][0]['text'] = re_text
#             # response.data.generations[0][0].text = self.extract_text(response)
#             return response
#         except asyncio.TimeoutError:
#             print("User message proposal request timed out")
#             return None
#         except Exception as e:
#             print(f"Error in usr_msg_proposal: {e}")
#             return None

# # 아래는 pydantic 테스트 ======================================================

# # from vllm import LLM, SamplingParams
# # from vllm.sampleing_params import GuidedDecodingParams
# # from pydantic import BaseModel, Field
# # import json

# # class UserMsgOutput(BaseModel):
# #     output: str = Field(
# #         min_length=10,
# #         max_length=50,
# #         description="Augment the user input into a detailed narrative."
# #     )

# # json_schema = UserMsgOutput.model_json_schema()
# # guided_decoding_params = GuidedDecodingParams(json=json_schema)
# # sampling_params = SamplingParams(guided_decoding=guided_decoding_params)

# # class OpenAIUsrMsgClient:
# #     def __init__(self, usr_msg, batch_handler):
# #         self.batch_handler = batch_handler
# #         self.usr_msg = str(usr_msg)
            

# #     async def usr_msg_proposal(self, max_tokens: int = 500):
# #         try:
# #             prompt = f"""
# #             [SYSTEM]
# #             You are a professional in business plan writing. You are provided with user input in the form of a sentence or paragraph. Your task is to write a narrative paragraph to assist in creating a business plan based on this input. Follow these instructions precisely:

# #             #### INSTRUCTIONS ####
# #             STEP 1. Read the user input carefully. The default language is Korean, but if the input is in another language (e.g., English), use that language for the output. If the input contains mixed languages, prioritize the language of the majority of the text.
# #             STEP 2. Identify and include key information from the user input, such as company name, brand name, products/services, and target customers, as keywords in the paragraph. Do not add information beyond what is provided.
# #             STEP 3. Develop the business plan narrative step-by-step using only the keywords and details from the user input. Do not expand the scope beyond the provided content.
# #             STEP 4. Write a paragraph of 1000 to 1500 characters to ensure the content is detailed and informative.
# #             STEP 5. Avoid repeating the same content to meet the character limit. Use varied expressions and vocabulary to enrich the narrative.
# #             STEP 6. Do not include repeated User-Assistant interactions or unnecessary filler. End the output naturally after completing the narrative.
# #             STEP 7. Ensure the text is free of typos and grammatical errors.
# #             STEP 8. Follow the gudied_decoding.

# #             [USER]
# #             user input = {self.usr_msg}
# #             """
# #             print(f"usr_msg: {self.usr_msg}")
# #             response = await asyncio.wait_for(
# #                 self.batch_handler.process_single_request({
# #                     "prompt": prompt,
# #                     "max_tokens": max_tokens,  # 1000~1500자 출력 위해 충분히 설정
# #                     "temperature": 0.7,
# #                     "top_p": 0.9,  # 더 자연스러운 출력 위해 조정
# #                     "repetition_penalty": 1.2,
# #                     "frequency_penalty": 1.0,
# #                     "n": 1,
# #                     "stream": False,
# #                     "logprobs": None
# #                 },
# #                                                           sampling_params=sampling_params,
# #                                                           request_id=0),
# #                 timeout=60
# #             )
# #             print("Result text of usr_msg : ", response.data.generations[0][0].text)
# #             print("Response of usr_msg : ", response)
            
# #             response.data.generations[0][0].text = self.extract_text(response)
# #             return response
# #         except asyncio.TimeoutError:
# #             print("User message proposal request timed out")
# #             return None
# #         except Exception as e:
# #             print(f"Error in usr_msg_proposal: {e}")
# #             return None


# # 위는 pydantic 테스트 ======================================================

#     # def extract_text(self, result):
#     #     print(f"result : {result}")
#     #     if result.success and 'generations' in result.data:
#     #         text = result.data['generations'][0][0]['text']
#     #         print(f"Extracted text: {text}")
#     #         # JSON 파싱 시도
#     #         json_data = self.extract_json(text)
#     #         if isinstance(json_data, dict) and "Output" in json_data:
#     #             return json_data["Output"]
#     #         # JSON이 아니면 정제된 텍스트 반환
#     #         return json_data if isinstance(json_data, str) else "텍스트 생성 실패"
#     #     return "텍스트 생성 실패"

#     # def extract_json(self, text):
#     #     text = re.sub(r'[\n\r\\\\/]', '', text, flags=re.DOTALL)
        
#     #     def clean_data(text):
#     #         headers_to_remove = [
#     #             "<|start_header_id|>system<|end_header_id|>",
#     #             "<|start_header_id|>SYSTEM<|end_header_id|>",
#     #             "<|start_header_id|>", "<|end_header_id|>",
#     #             "<|start_header_id|>user<|end_header_id|>",
#     #             "<|start_header_id|>assistant<|end_header_id|>",
#     #             "<|eot_id|><|start_header_id|>ASSISTANT_EXAMPLE<|end_header_id|>",
#     #             "<|eot_id|><|start_header_id|>USER_EXAMPLE<|end_header_id|>",
#     #             "<|eot_id|><|start_header_id|>USER<|end_header_id|>",
#     #             "<|eot_id|>",
#     #             "<|eot_id|><|start_header_id|>ASSISTANT<|end_header_id|>",
#     #             "ASSISTANT",
#     #             "USER",
#     #             "SYSTEM",
#     #             "<|end_header_id|>",
#     #             "<|start_header_id|>",
#     #             "ASSISTANT_EXAMPLE",
#     #             "USER_EXAMPLE",
#     #             "[]",
#     #             "Output",
#     #             "output",
#     #             "="
#     #         ]
#     #         cleaned_text = text
#     #         for header in headers_to_remove:
#     #             cleaned_text = cleaned_text.replace(header, '')
#     #         pattern = r'<\|.*?\|>'
#     #         cleaned_text = re.sub(pattern, '', cleaned_text)
#     #         return cleaned_text.strip()

#     #     text = clean_data(text)
#     #     # JSON 객체를 찾음
#     #     json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', text, re.DOTALL)
#     #     if json_match:
#     #         json_str = json_match.group()
#     #         open_braces = json_str.count('{')
#     #         close_braces = json_str.count('}')
#     #         if open_braces > close_braces:
#     #             json_str += '}' * (open_braces - close_braces)
#     #         try:
#     #             return json.loads(json_str)
#     #         except json.JSONDecodeError:
#     #             try:
#     #                 return json.loads(json_str.replace("'", '"'))
#     #             except json.JSONDecodeError:
#     #                 return text  # JSON 파싱 실패 시 정제된 텍스트 반환
#     #     return text  # JSON 없으면 정제된 텍스트 반환