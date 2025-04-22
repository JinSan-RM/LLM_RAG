import asyncio
import logging
import json
import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OpenAIProposalClient:
    def __init__(self, pdf_content: str, batch_handler):
        self.pdf_content = pdf_content or ""
        self.batch_handler = batch_handler

    async def send_request(self, sys_prompt: str, usr_prompt: str, temperature:float, top_p:float, max_tokens: int = 100) -> str:

        model = "/usr/local/bin/models/gemma-3-4b-it"
        try:
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                    "model": model,
                    "sys_prompt": sys_prompt,
                    "usr_prompt": usr_prompt,
                    # "extra_body": extra_body,
                    "max_tokens": max_tokens,
                    "temperature": temperature,  # 안정성 우선
                    "top_p": top_p,
                    "n": 1,
                    "stream": False,
                    "logprobs": None,
                }, request_id=0),
                timeout=120  # 타임아웃 설정
            )
            return response
        except asyncio.TimeoutError:
            print("[ERROR] Request timed out")
            return "Error: Request timed out after 120 seconds"
        except Exception as e:
            print(f"[ERROR] Unexpected error: {str(e)}")
            return f"Error: {str(e)}"

    async def generate_proposal(self, max_tokens: int = 500, temperature = 0.7, top_p = 0.9) -> dict:
        try:
            sys_prompt = """
 
            You are an expert at crafting business plan. 
            Create a concise business plan by using the pdf_content.
            Mix two inputs and proceed with the below.
            Only to add supporting details, if the informartion is not enough.

            #### INSTRUCTIONS ####
            1. It must drive the narrative and tone.
            2. Include these seven elements, filling gaps creatively based on the pdf_content:
                - Business Item (what we offer)
                - Slogan (short and catchy)
                - Target Customers (who we serve)
                - Core Value (why choose us)
                - Features (key benefits)
                - Business Model (how we profit)
                - Marketing Strategy (how we reach customers)
            3. Write a paragraph of 500 to 800 characters to ensure the content is detailed and informative.
            4. Include all the key information (numbers, examples, etc.) without missing anything.
            5. 반드시 **한국어**로 작성해.
            """
    
            usr_prompt = f" pdf_content:{self.pdf_content}"
 
            usr_prompt = f"PDF 내용: {self.pdf_content}"
            
            result = await self.send_request(
                sys_prompt=sys_prompt, 
                usr_prompt=usr_prompt, 
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
                )            
            
            if result and hasattr(result, 'data') and 'generations' in result.data:
                text = result.data['generations'][0][0]['text'].replace("\n", " ")
                result.data['generations'][0][0]['text'] = text
                return result.data
            return {"error": "Proposal 생성 오류"}
        except Exception as e:
            logger.error(f"Proposal 생성 오류: {str(e)}")
            return {"error": str(e)}
        
    async def consolidate_proposals(self, proposals: List[str], max_tokens: int = 800, temperature = 0.3, top_p = 0.3) -> dict:
        """PDF가 2개 이상일 때 Proposal을 통합"""
        try:
            sys_prompt = """
            You are an expert at crafting business plans. 
            Consolidate proposals generated from multiple PDFs. The number of Proposals can be between 1 and 3.
            Mix inputed Proposals and proceed with the below.
            Only add supporting details if the information is not enough.

            #### INSTRUCTIONS ####
            1. Never create a hypothetical proposal, and only work within the input you are given.
            2. It must drive the narrative and tone.
            3. Include these seven elements, filling gaps creatively based on the inputed Proposals:
                - Business Item (what we offer)
                - Slogan (short and catchy)
                - Target Customers (who we serve)
                - Core Value (why choose us)
                - Features (key benefits)
                - Business Model (how we profit)
                - Marketing Strategy (how we reach customers)
            4. Write a paragraph of 800 to 1000 characters to ensure the content is detailed and informative.
            5. Include all the key information (numbers, examples, etc.) without missing anything.
            6. 반드시 **한국어**로 작성해.
            """
            usr_prompt = "\n".join([f"Proposal_{i+1}: {p}" for i, p in enumerate(proposals)])
            
            result = await self.send_request(
                sys_prompt=sys_prompt, 
                usr_prompt=usr_prompt, 
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
                )            
            
            if result and hasattr(result, 'data') and 'generations' in result.data:
                text = result.data['generations'][0][0]['text'].replace("\n", " ")
                result.data['generations'][0][0]['text'] = text
                return result.data
            return {"error": "Proposal 통합 오류"}
        except Exception as e:
            logger.error(f"Proposal 통합 오류: {str(e)}")
            return {"error": str(e)}
        
# class OpenAIPDFSummaryClient:
#     def __init__(self, pdf_data: str, batch_handler):
#         self.pdf_data = pdf_data or ""
#         self.batch_handler = batch_handler
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=3000,  # 청크당 최대 문자 수
#             chunk_overlap=1000,  # 중복 문자 수 (문맥 유지)
#             length_function=len,
#             separators=["\n\n", "\n", ".", " ", ""]
#         )

#     @staticmethod
#     def extract_json(text):
#         """불완전한 JSON 문자열을 파싱하거나 보정"""
#         try:
#             text = text.strip()
#             if text.endswith(','):
#                 text = text[:-1]
#             if not text.endswith(']'):
#                 text += ']'
#             return json.loads(text)
#         except json.JSONDecodeError:
#             logger.error(f"JSON 파싱 실패: {text[:50]}")
#             return None

#     async def summarize_chunk(self, chunk: str) -> dict:
#         """개별 청크를 Chain of Density 방식으로 요약"""
#         try:
#             sys_prompt = """
#             PDF 텍스트를 한국어로 500~700자 이내로 요약하세요.
#             랜딩페이지용으로 핵심 정보만 간결히 전달하며, 반복 내용은 제거하고, 불필요한 태그나 메타데이터는 제외하세요.
#             모든 주요 사실(숫자, 사례, 전략 등)을 누락 없이 포함하고 기획서 형태로 생성하세요.
#             """
#             usr_prompt = f"""
#             {chunk}
#             """
#             response = await asyncio.wait_for(
#                 self.batch_handler.process_single_request({
#                     "sys_prompt": sys_prompt,
#                     "usr_prompt": usr_prompt,
#                     "max_tokens": 1000,
#                     "temperature": 0.2,
#                     "top_p": 0.9
#                 }, request_id=0),
#                 timeout=120
#             )

#             if response is None or not hasattr(response, 'data') or 'generations' not in response.data:
#                 logger.error("청크 요약 API 응답 오류")
#                 return {"error": "API 응답 오류: 유효한 데이터 없음"}

#             text = response.data['generations'][0][0]['text'].replace("\n", " ")
#             response.data['generations'][0][0]['text'] = text
#             logger.debug(f"청크 요약 완료 - 결과 길이: {len(text)}")
#             return response.data
#         except asyncio.TimeoutError:
#             logger.error("청크 요약 타임아웃")
#             return {"error": "청크 요약 타임아웃"}
#         except Exception as e:
#             logger.error(f"청크 요약 오류: {str(e)}", exc_info=True)
#             return {"error": str(e)}

#     async def summarize_pdf(self, pdf_content: str) -> dict:
#             try:
#                 if not pdf_content.strip():
#                     return {"error": "PDF 내용이 비어 있음"}

#                 # PDF를 2개 청크로 분할 (약 2500자씩)
#                 mid_point = len(pdf_content) // 2
#                 chunks = [pdf_content[:mid_point], pdf_content[mid_point:]]
#                 logger.debug(f"PDF 청킹 완료 - 청크 수: {len(chunks)}, 크기: {len(chunks[0])}, {len(chunks[1])}")

#                 # 2개 청크 병렬 요약
#                 tasks = [self.summarize_chunk(chunk) for chunk in chunks]
#                 chunk_summaries = await asyncio.gather(*tasks, return_exceptions=True)

#                 # 요약 결과 통합
#                 all_summaries = [r['generations'][0][0]['text'] for r in chunk_summaries if isinstance(r, dict) and "error" not in r]
#                 if not all_summaries:
#                     return {"error": "모든 청크 요약 실패"}
#                 combined_summary = " ".join(all_summaries)
#                 return {"generations": [[{"text": combined_summary}]]}
#             except Exception as e:
#                 logger.error(f"PDF 요약 오류: {str(e)}")
#                 return {"error": str(e)}
            


        
# class OpenAIProposalClient:
#     def __init__(self, summary_text: str, batch_handler):
#         self.summary_text = summary_text
#         self.batch_handler = batch_handler

#     async def generate_proposal(self) -> dict:
#         try:
#             sys_prompt = """
#             PDF 요약 데이터를 기반으로 랜딩페이지용 Proposal을 700~1000자로 작성하세요.
#             모든 핵심 정보(숫자, 사례 등)를 누락 없이 포함하고, 20~30대 타겟에 맞춘 캐주얼하고 매력적인 톤으로 작성하세요.
#             """
#             usr_prompt = f"summary_text: {self.summary_text}"
#             response = await asyncio.wait_for(
#                 self.batch_handler.process_single_request({
#                     "sys_prompt": sys_prompt,
#                     "usr_prompt": usr_prompt,
#                     "max_tokens": 500,
#                     "temperature": 0.3,
#                     "top_p": 0.9
#                 }, request_id=0),
#                 timeout=120
#             )
#             if response and hasattr(response, 'data') and 'generations' in response.data:
#                 text = response.data['generations'][0][0]['text'].replace("\n", " ")
#                 response.data['generations'][0][0]['text'] = text
#                 return response.data
#             return {"error": "Proposal 생성 오류"}
#         except Exception as e:
#             logger.error(f"Proposal 생성 오류: {str(e)}")
#             return {"error": str(e)}
# ==========================================================
# test
# ==========================================================
# import asyncio
# import json
# import re
# from typing import List

# import time
# import textwrap

# from langchain import hub
# from langchain.chat_models import ChatOpenAI
# from langchain.output_parsers.json import SimpleJsonOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# class OpenAIPDFSummaryClient:
#     def __init__(self, pdf_data: str, batch_handler):
#         self.pdf_data = pdf_data
#         self.batch_handler = batch_handler

#     @staticmethod
#     def extract_json(text):
#         try:
#             # 불완전한 JSON 문자열 완성 시도
#             text = text.strip()
#             if text.endswith(','):
#                 text = text[:-1]
#             if not text.endswith('}'):
#                 text += '}'
#             return json.loads(text)
#         except json.JSONDecodeError:
#             print("JSON 파싱 실패")
#             return None

# # Chain of Density : https://smith.langchain.com/hub/whiteforest/chain-of-density-prompt?organizationId=b2a772f8-ac29-42f8-8750-0f1a4f6cfff0
        
        
#     def chunking_text(self, content: str, chunk_size: int = 200, chunk_overlap: int = 50) -> List[str]:
    
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,   # 한 Chunk의 최대 길이 글자 수를 의미미
#             chunk_overlap=chunk_overlap  # Chunk 간 오버랩 (중복 내용 포함)
#         )
#         text_chunks = text_splitter.split_text(content)

#         return text_chunks

#     async def summarize_chunked_texts_with_CoD(self, pdf_texts: str, chunk_size: int, chunk_overlap: int) -> str:
#         try:
#             # 텍스트 분할 작업
#             chunk_texts = self.chunking_text(content=pdf_texts, 
#                                             chunk_size=chunk_size, 
#                                             chunk_overlap=chunk_overlap)
            
#             if not chunk_texts:
#                 print("분할 결과가 없습니다.")
#                 return ""
                
#             tasks = [self.summarize_text_with_CoD(chunk) for chunk in chunk_texts]
#             print(f"tasks : {tasks}")
#             results = []
#             for completed_task in asyncio.as_completed(tasks):
#                 result = await completed_task
#                 # 완료되는 대로 결과 처리
#                 results.append(result)
#             # results = await asyncio.gather(*tasks, return_exceptions=True)
#             print(f"results : {results}")
#             # 요약 추출 병합
#             all_summaries = []
#             for idx, result in enumerate(results):
#                 if isinstance(result, Exception):
#                     continue
                    
#                 try:
#                     print(f"result : {result}")
#                     text_result = result.data['generations'][0][0]['text']
#                     text_result = re.sub(r'[\x00-\x1F\x7F]', '', text_result)  # 제어 문자 제거
                    
#                     chunk_summaries = re.findall(r'"denser_summary"\s*:\s*"([^"]+)"', text_result)
#                     if chunk_summaries:
#                         all_summaries.extend(chunk_summaries)
#                     else:
#                         print(f"chunk 데이터 {idx}에서 요약 추출 실패")
                        
#                 except Exception as e:
#                     print(f"데이터 {idx} 결과 처리 오류: {str(e)}")
#                     continue

#             if not all_summaries:
#                 print("모든 데이터 요약에 실패했습니다.")
#                 return ""

#             final_combined_string = " ".join(all_summaries)

#             # 최종 요약
#             final_summary = await self.summarize_text_with_CoD(
#                 content=final_combined_string,
#                 content_category="business report",
#                 entity_range=2,
#                 max_words=1000,
#                 iterations=2
#             )
#             try:
#                 text_result = final_summary.data['generations'][0][0]['text']
#                 final_summaries = re.findall(r'"denser_summary"\s*:\s*"([^"]+)"', text_result)
#                 final_text = " ".join(final_summaries)

#                 if final_text:
#                     final_summary.data['generations'][0][0]['text'] = final_text

#             except Exception as e:
#                 print(f"최종 요약 처리 오류: {str(e)}")

#             return final_summary

#         except Exception as e:
#             print(f"요약 프로세스 오류: {str(e)}")
#             return ""


#     async def summarize_text_with_CoD(self, content: str, content_category: str = "business report", entity_range:int = 3, max_words:int = 80, iterations:int = 3) -> str:
        
#         sys_prompt = f"""
#             As an expert copy-writer, you will write increasingly concise, entity-dense summaries of the user provided {content_category}. The initial summary should be under {max_words} words and contain {entity_range} informative Descriptive Entities from the {content_category}.

#             A Descriptive Entity is:
#             - Relevant: to the main story.
#             - Specific: descriptive yet concise (5 words or fewer).
#             - Faithful: present in the {content_category}.
#             - Anywhere: located anywhere in the {content_category}.

#             # Your Summarization Process
#             - Read through the {content_category} and the all the below sections to get an understanding of the task.
#             - Pick {entity_range} informative Descriptive Entities from the {content_category} (";" delimited, do not add spaces).
#             - In your output JSON list of dictionaries, write an initial summary of max {max_words} words containing the Entities.

#             Then, repeat the below 2 steps {iterations} times:

#             - Step 1. In a new dict in the same list, identify {entity_range} new informative Descriptive Entities from the {content_category} which are missing from the previously generated summary.
#             - Step 2. Write a new, denser summary of identical length which covers every Entity and detail from the previous summary plus the new Missing Entities.

#             A Missing Entity is:
#             - An informative Descriptive Entity from the {content_category} as defined above.
#             - Novel: not in the previous summary.

#             # Guidelines
#             - The first summary should be long (max {max_words} words) yet highly non-specific, containing little information beyond the Entities marked as missing. Use overly verbose language and fillers (e.g., "this {content_category} discusses") to reach ~{max_words} words.
#             - Make every word count: re-write the previous summary to improve flow and make space for additional entities.
#             - Make space with fusion, compression, and removal of uninformative phrases like "the {content_category} discusses".
#             - The summaries should become highly dense and concise yet self-contained.
#             - Missing entities can appear anywhere in the new summary.
#             - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
#             - You're finished when your JSON list has 1+{iterations} dictionaries of increasing density.

#             # IMPORTANT
#             - Remember, to keep each summary to max {max_words} words.
#             - Never remove Entities or details. Only add more from the {content_category}.
#             - Do not discuss the {content_category} itself, focus on the content: informative Descriptive Entities, and details.
#             - Remember, if you're overusing filler phrases in later summaries, or discussing the {content_category} itself, not its contents, choose more informative Descriptive Entities and include more details from the {content_category}.
#             - Answer with a minified JSON list of dictionaries with keys "missing_entities" and "denser_summary".
#             - 출력은 반드시 **한국어**로 해.

#             """
#             # - You now have `[{{"missing_entities": "...", "denser_summary": "..."}}]`
        
#             # ## Example output
#             # [{{"missing_entities": "ent1;ent2", "denser_summary": "<vague initial summary with entities 'ent1','ent2'>"}}, {{"missing_entities": "ent3", "denser_summary": "denser summary with 'ent1','ent2','ent3'"}}, ...]
#         usr_prompt = f"""
#         {content_category}:
#         {content}
#         """

#         try:

#             extra_body = {
#                         "guided_json": {
#                             "type": "array",  # 리스트로 설정
#                             "items": {  # 배열의 각 항목 정의
#                                 "type": "object",
#                                 "properties": {
#                                     "missing_entities": {"type": "string"},
#                                     "denser_summary": {"type": "string"}
#                                 },
#                                 "required": ["missing_entities", "denser_summary"]
#                             },
#                             "minItems": 2,  # 최소 4개 항목
#                             "maxItems": 4   # 최대 4개 항목 (정확히 4개 강제)
#                         }
#                     }
#             start = time.time()
#             result = await asyncio.wait_for(
#                 self.batch_handler.process_single_request({
#                     "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
#                     "sys_prompt": sys_prompt,
#                     "usr_prompt": usr_prompt,
#                     "extra_body": extra_body,
#                     "max_tokens": 1000,
#                     "temperature": 0.1,
#                     "top_p": 0.3}, request_id=0),
#                 timeout=100
#             )
#             end = time.time()
#             now = time.time()
#             print(f"COD result {end - start} {now} : {result}")
#             if result.success:
#             # Extract the summary from the response
#                 response = result
#                 return response  # Return the generated summary

#             else:
#                 # Log the error details if the request failed
#                 print(f"[ERROR] Summary generation failed with error: {result.error}")
#                 return f"[ERROR] Summary generation failed with error: {result.error}"

#         except Exception as e:
#             # Log any unexpected errors
#             print(f"[ERROR] Unexpected error during summarization: {str(e)}")
#             return f"[ERROR] Unexpected error during summarization: {str(e)}"

#     async def generate_proposal(self, summary: str):
#         try:
#             sys_prompt = f"""
#                 You are a professional in business plan writing. 
#                 You are provided with summarized pdf input from user.
#                 Your task is to write a narrative paragraph to assist in creating a business plan based on the input. 
#                 Follow these instructions precisely:
                
#                 Ensure the script is logically structured with clear sections and transitions. Maintain all core information without summarizing.
#                 Avoid the hallucination and frame the script as an independent narrative.
                
#                 #### INSTRUCTIONS ####
                
#                 1. Identify and include key information from the user input, such as below. If the usr_input is not enough, you can fill it yourself.
#                     - BUSINESS ITEM: Specific product or service details
#                     - SLOGAN OR CATCH PHRASE: A sentence expressing the company's main vision or ideology
#                     - TARGET CUSTOMERS: Characteristics and needs of the major customer base
#                     - CORE VALUE PROPOSITION: Unique value provided to customers
#                     - PRODUCT AND SERVICE FEATURES: Main functions and advantages
#                     - BUSINESS MODEL: Processes that generate profits by providing differentiated value
#                     - PROMOTION AND MARKETING STRATEGY: How to introduce products or services to customers            
#                 2. Develop the business plan narrative step-by-step using only the keywords and details from the user input. Do not expand the scope beyond the provided content.
#                 3. Write a paragraph of 1000 to 1500 characters to ensure the content is detailed and informative.
#                 4. Avoid repeating the same content to meet the character limit. Use varied expressions and vocabulary to enrich the narrative.
#                 5. 출력은 반드시 **한국어**로 해.            
#                 """
            
#             usr_prompt = f"""
#             {summary}
#             """

#             response = await asyncio.wait_for(
#                 self.batch_handler.process_single_request({
#                         "sys_prompt": sys_prompt,
#                         "usr_prompt": usr_prompt,
#                         "max_tokens": 1000,
#                         "temperature": 0.7,
#                         "top_p": 0.3,
#                         "n": 1,
#                         "stream": False,
#                         "logprobs": None
#                     }, request_id=0),
#                 timeout=100  # 적절한 타임아웃 값 설정
#             )
#             print("generate_proposal result : ", response)
#             return response
#         except Exception as e:
#             print(f"제안서 생성 중 예상치 못한 오류: {str(e)}")
#             return ""
        
# #========================================
# #   기존 버전
# #========================================

#     async def summarize_text(self, pdf_str: str, max_tokens: int = 1500) -> str:
#         try:
            
#             # NOTE : 이 부분은 나중에는 연산을 넣어서 판단하면 될듯
#             is_korean = any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in pdf_str)
#             output_language = "Korean" if is_korean else "English"            
#             # STEP 1. Read the user_input carefully. The default language is {output_language}
#             # prompt = f"""
#             # [SYSTEM]
#             # You are a professional in business plan writing from business plan contents in PDF.
#             # Your task is to write a narrative paragraph to assist in creating a business plan based on user_input. 
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
#             # STEP 3. Summarize a paragraph of 1000 to 1500 characters to ensure the content is detailed and informative.      
#             # STEP 4. Ensure the text is free of typos and grammatical errors.
#             # STEP 5. Output only the final business plan narrative text. Do not include tags (e.g., [SYSTEM], <|eot_id|>), JSON formatting (e.g., {{Output: "..."}}), or any metadata in the output.            
#             # STEP 6. 출력은 반드시 **한국어**로 해.
            
#             # [USER]
#             # user_input = {pdf_str}
#             # """
#             sys_prompt = f"""
#             You are a professional in business plan writing from business plan contents in PDF.
#             Your task is to write a narrative paragraph to assist in creating a business plan based on user_input. 
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
#             STEP 3. Summarize a paragraph of 1000 to 1500 characters to ensure the content is detailed and informative.      
#             STEP 4. Ensure the text is free of typos and grammatical errors.
#             STEP 5. Output only the final business plan narrative text. Do not include tags (e.g., [SYSTEM], <|eot_id|>), JSON formatting (e.g., {{Output: "..."}}), or any metadata in the output.            
#             STEP 6. 출력은 반드시 **한국어**로 해.
#             """
            
#             usr_prompt = f"""
#             user_input = {pdf_str}
#             """

#             response = await asyncio.wait_for(
#                 self.batch_handler.process_single_request({
#                     # "prompt": prompt,
#                     "sys_prompt": sys_prompt,
#                     "usr_prompt": usr_prompt,
#                     "max_tokens": max_tokens,
#                     "temperature": 0.7,
#                     "top_p": 0.3,
#                     "n": 1,
#                     "stream": False,
#                     "logprobs": None
#                 }, request_id=0),
#                 timeout=240
#             )
#             # 응답 처리
#             if response.success and response.data:
#                 # extracted_text = self.extract_text(response)
#                 re_text = response.data['generations'][0][0]['text']
#                 re_text = re_text.replace("\n", " ")
#                 response.data['generations'][0][0]['text'] = re_text
#                 # 상위 호출과 호환성을 위해 generations 구조로 변환
#                 return response
#             else:
#                 print(f"[ERROR] Summary generation failed: {response.error}")
#                 response.data = {"generations": [{"text": "텍스트 생성 실패"}]}
#                 return response

#         except asyncio.TimeoutError:
#             print("요약 요청 시간 초과")
#             response = type('MockResponse', (), {'success': False, 'data': {"generations": [{"text": "요약 요청 시간 초과"}]}})()
#             return response
#         except Exception as e:
#             print(f"요약 중 예상치 못한 오류: {str(e)}")
#             response = type('MockResponse', (), {'success': False, 'data': {"generations": [{"text": f"오류: {str(e)}"}]}})()
#             return response

#     def extract_text(self, result):
#         if result.success and result.data.generations:
#             json_data = self.extract_json(result.data.generations[0][0].text)
#             result.data.generations[0][0].text = json_data
#             return result
#         return "텍스트 생성 실패"

#     def extract_json(self, text):
#         text = re.sub(r'[\n\r\\\\/]', '', text, flags=re.DOTALL)
        
#         def clean_data(text):
#             headers_to_remove = [
#                 "<|start_header_id|>system<|end_header_id|>",
#                 "<|start_header_id|>SYSTEM<|end_header_id|>",
#                 "<|start_header_id|>", "<|end_header_id|>",
#                 "<|start_header_id|>user<|end_header_id|>",
#                 "<|start_header_id|>assistant<|end_header_id|>",
#                 "<|eot_id|><|start_header_id|>ASSISTANT_EXAMPLE<|end_header_id|>",
#                 "<|eot_id|><|start_header_id|>USER_EXAMPLE<|end_header_id|>",
#                 "<|eot_id|><|start_header_id|>USER<|end_header_id|>",
#                 "<|eot_id|>",
#                 "<|eot_id|><|start_header_id|>ASSISTANT<|end_header_id|>",
#                 "[ASSISTANT]",
#                 "[USER]",
#                 "[SYSTEM]",
#                 "<|end_header_id|>",
#                 "<|start_header_id|>"
#                 "ASSISTANT_EXAMPLE",
#                 "USER_EXAMPLE",
#                 "Output",
#                 "output",
#                 "=",
#                 "{",
#                 "}",
#                 ":"
#             ]
#             cleaned_text = text
#             for header in headers_to_remove:
#                 cleaned_text = cleaned_text.replace(header, '')
#             pattern = r'<\|.*?\|>'
#             cleaned_text = re.sub(pattern, '', cleaned_text)
#             return cleaned_text.strip()
        
#         text = clean_data(text)
        
#         # JSON 객체를 찾음
#         json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', text, re.DOTALL)
#         if json_match:
#             json_str = json_match.group()
#             open_braces = json_str.count('{')
#             close_braces = json_str.count('}')
#             if open_braces > close_braces:
#                 json_str += '}' * (open_braces - close_braces)
#             try:
#                 return json.loads(json_str)
#             except json.JSONDecodeError:
#                 try:
#                     return json.loads(json_str.replace("'", '"'))
#                 except json.JSONDecodeError:
#                     return text  # JSON 파싱 실패 시 원본 텍스트 반환
#         else:
#             # JSON이 없으면 정리된 텍스트 반환
#             return text.strip()