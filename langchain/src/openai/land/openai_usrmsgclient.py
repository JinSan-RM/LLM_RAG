from langchain.prompts import PromptTemplate
import asyncio
import re
import json

class OpenAIUsrMsgClient:
    def __init__(self, usr_msg, batch_handler):
        self.batch_handler = batch_handler
        self.usr_msg = str(usr_msg)
            

    async def usr_msg_proposal(self, max_tokens: int = 500):
        try:
            prompt = f"""
            [SYSTEM]
            You are a professional in business plan writing. You are provided with user input in the form of a sentence or paragraph. Your task is to write a narrative paragraph to assist in creating a business plan based on this input. Follow these instructions precisely:

            #### INSTRUCTIONS ####
            STEP 1. Read the user input carefully. The default language is Korean, but if the input is in another language (e.g., English), use that language for the output. If the input contains mixed languages, prioritize the language of the majority of the text.
            STEP 2. Identify and include key information from the user input, such as company name, brand name, products/services, and target customers, as keywords in the paragraph. Do not add information beyond what is provided.
            STEP 3. Develop the business plan narrative step-by-step using only the keywords and details from the user input. Do not expand the scope beyond the provided content.
            STEP 4. Write a paragraph of 1000 to 1500 characters to ensure the content is detailed and informative.
            STEP 5. Avoid repeating the same content to meet the character limit. Use varied expressions and vocabulary to enrich the narrative.
            STEP 6. Do not include repeated User-Assistant interactions or unnecessary filler. End the output naturally after completing the narrative.
            STEP 7. Ensure the text is free of typos and grammatical errors.
            STEP 8. Output only the final business plan narrative text. Do not include tags (e.g., [SYSTEM], <|eot_id|>), JSON formatting (e.g., {{Output: "..."}}), or any metadata in the output.

            [USER]
            user input = {self.usr_msg}
            """
            print(f"usr_msg: {self.usr_msg}")
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                    "prompt": prompt,
                    "max_tokens": max_tokens,  # 1000~1500자 출력 위해 충분히 설정
                    "temperature": 0.7,
                    "top_p": 0.9,  # 더 자연스러운 출력 위해 조정
                    "repetition_penalty": 1.2,
                    "frequency_penalty": 1.0,
                    "n": 1,
                    "stream": False,
                    "logprobs": None
                }, request_id=0),
                timeout=60
            )
            response.data.generations[0][0].text = self.extract_text(response)
            print("=========== USR_MSG_CLIENT ===========")
            print(f"extracted_text_usr_msg_client : {response.data.generations[0][0].text}")
            print(f"All_response_of_usr_msg_client : {response}")
            print("======================================")
            return response
        except asyncio.TimeoutError:
            print("User message proposal request timed out")
            return None
        except Exception as e:
            print(f"Error in usr_msg_proposal: {e}")
            return None

    def extract_text(self, result):
        print(f"result final: {result}")
        if result.success and result.data.generations:
            text = result.data.generations[0][0].text
            # JSON 파싱 시도
            json_data = self.extract_json(text)
            if isinstance(json_data, dict) and "Output" in json_data:
                return json_data["Output"]
            # JSON이 아니면 정제된 텍스트 반환
            return json_data if isinstance(json_data, str) else "텍스트 생성 실패"
        return "텍스트 생성 실패"

    def extract_json(self, text):
        text = re.sub(r'[\n\r\\\\/]', '', text, flags=re.DOTALL)
        
        def clean_data(text):
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
                "<|start_header_id|>",
                "ASSISTANT_EXAMPLE",
                "USER_EXAMPLE",
                "[]"
            ]
            cleaned_text = text
            for header in headers_to_remove:
                cleaned_text = cleaned_text.replace(header, '')
            pattern = r'<\|.*?\|>'
            cleaned_text = re.sub(pattern, '', cleaned_text)
            return cleaned_text.strip()

        text = clean_data(text)
        # JSON 객체를 찾음
        json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    return json.loads(json_str.replace("'", '"'))
                except json.JSONDecodeError:
                    return text  # JSON 파싱 실패 시 정제된 텍스트 반환
        return text  # JSON 없으면 정제된 텍스트 반환

# from langchain.prompts import PromptTemplate
# import asyncio
# import json
# from pydantic import BaseModel, Field
# from typing import Any, Optional

# # ✅ 1️⃣ 요청 데이터 검증 모델
# class UsrMsgRequest(BaseModel):
#     model: str = Field(..., description="사용할 LLM 모델 경로")
#     usr_msg: str = Field(..., description="사용자의 입력 메시지")
#     max_tokens: int = Field(500, ge=100, le=2000, description="최대 토큰 수 (기본값 500)")

# # ✅ 2️⃣ VLLM 요청 데이터 검증 모델
# class VLLMRequest(BaseModel):
#     prompt: str
#     max_tokens: int = 500
#     temperature: float = 0.7
#     top_p: float = 0.9
#     repetition_penalty: float = 1.2
#     frequency_penalty: float = 1.0
#     n: int = 1
#     stream: bool = False
#     logprobs: Any = None

# # ✅ 3️⃣ 모델 응답 검증 및 JSON 처리
# class UsrMsgResponse(BaseModel):
#     text: str = Field(..., description="생성된 비즈니스 계획 내러티브")

# import json

# class OpenAIUsrMsgClient:
#     def __init__(self, request_data: Any, batch_handler: Any):
#         # ✅ request_data가 문자열이면 JSON으로 변환
#         if isinstance(request_data, str):
#             try:
#                 request_data = json.loads(request_data)  # JSON 파싱
#             except json.JSONDecodeError:
#                 raise ValueError("Invalid JSON string provided as request_data.")

#         # ✅ request_data가 리스트면 첫 번째 요소 선택
#         if isinstance(request_data, list) and len(request_data) > 0:
#             request_data = request_data[0]

#         # ✅ request_data가 dict면 Pydantic 객체로 변환
#         if isinstance(request_data, dict):
#             request_data = UsrMsgRequest(**request_data)  

#         # ✅ request_data가 올바른 타입인지 확인
#         if not isinstance(request_data, UsrMsgRequest):
#             raise ValueError(f"Invalid input type: {type(request_data)}. Expected dict or UsrMsgRequest.")

#         self.request_data = request_data
#         self.batch_handler = batch_handler


#     # ✅ 4️⃣ JSON 기반 프롬프트 생성
#     def generate_prompt(self) -> str:
#         return f"""
#         [SYSTEM]
#         You are a professional in business plan writing.
#         You are provided with user input in the form of a sentence or paragraph.
#         Your task is to write a narrative paragraph to assist in creating a business plan based on this input.
#         Follow these instructions precisely:

#         #### INSTRUCTIONS ####
#         STEP 1. Read the user input carefully. The default language is Korean, but if the input is in another language (e.g., English), use that language for the output.
#         STEP 2. Identify and include key information from the user input, such as company name, brand name, products/services, and target customers, as keywords in the paragraph.
#         STEP 3. Develop the business plan narrative step-by-step using only the keywords and details from the user input.
#         STEP 4. Write a paragraph of 1000 to 1500 characters.
#         STEP 5. Use varied expressions and vocabulary to enrich the narrative.
#         STEP 6. Ensure the text is free of typos and grammatical errors.
#         STEP 7. Output only the final business plan narrative text. Do not include tags (e.g., [SYSTEM]), JSON formatting, or any metadata.

#         [USER]
#         user input = {self.request_data.usr_msg}
#         """

#     async def usr_msg_proposal(self):
#         try:
#             # ✅ 프롬프트 생성
#             prompt = self.generate_prompt()
#             vllm_request = VLLMRequest(prompt=prompt, max_tokens=self.request_data.max_tokens)

#             # ✅ 응답 받기
#             response = await asyncio.wait_for(
#                 self.batch_handler.process_single_request(vllm_request.dict(), request_id=0),
#                 timeout=60
#             )

#             # ✅ response가 dict인지 확인
#             if isinstance(response, dict):
#                 response_data = response.get("data") or response  # ✅ response.data 대신 get 사용
#             else:
#                 response_data = response.data  # ✅ 기존 구조

#             # ✅ 응답 검증 및 JSON 처리
#             extracted_text = self.extract_text(response_data)

#             return {"type": "usr_msg_result", "result": UsrMsgResponse(text=extracted_text).dict()}

#         except asyncio.TimeoutError:
#             return {"type": "error", "message": "텍스트 생성 실패: 시간 초과"}
#         except Exception as e:
#             return {"type": "error", "message": f"텍스트 생성 실패: {str(e)}"}


#     def extract_text(self, result) -> str:
#         """응답에서 텍스트를 추출하고 JSON 파싱을 시도"""
#         print(f"Raw Response: {result}")

#         if hasattr(result, "data") and result.data.generations:
#             text = result.data.generations[0][0].text

#             # JSON 파싱 시도
#             json_data = self.extract_json(text)
#             if isinstance(json_data, dict) and "Output" in json_data:
#                 return json_data["Output"]

#             return json_data if isinstance(json_data, str) else "텍스트 생성 실패"

#         return "텍스트 생성 실패"

#     def extract_json(self, text: str) -> Optional[dict]:
#         """텍스트가 JSON 형식인지 검사하고 변환"""
#         try:
#             return json.loads(text)
#         except json.JSONDecodeError:
#             return text  # JSON 형식이 아니면 원래 텍스트 반환

