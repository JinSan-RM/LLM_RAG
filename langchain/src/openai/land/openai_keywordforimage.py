# import json
# import re
# import asyncio
# from src.utils.batch_handler import BatchRequestHandler


# class OpenAIKeywordClient:
#     def __init__(self, batch_handler: BatchRequestHandler):
#         self.batch_handler = batch_handler

#     async def send_request(self, prompt: str) -> str:
#         response = await asyncio.wait_for(
#             self.batch_handler.process_single_request({
#                 "prompt": prompt,
#                 "max_tokens": 100,
#                 "temperature": 0.1,
#                 "top_p": 1.0,
#                 "n": 1,
#                 "stream": False,
#                 "logprobs": None
#             }, request_id=0),
#             timeout=60  # 적절한 타임아웃 값 설정
#         )
#         return response

#     async def process_menu_data(self, menu_data: str) -> list:
#         try:
#             print(f"menu_data : {menu_data}")
#             json_match = re.search(r"\[.*\]", menu_data, re.DOTALL)
#             if not json_match:
#                 raise ValueError("JSON 형식을 찾을 수 없습니다.")
#             json_text = json_match.group()
#             json_text = json_text.replace("'", '"').replace("\n", "").strip()
#             json_data = json.loads(json_text)
#             return json_data
#         except json.JSONDecodeError as e:
#             print(f"JSON 파싱 실패: {e}")
#             raise RuntimeError("menu_data의 형식이 올바르지 않습니다.") from e

#     async def section_keyword_recommend(self, context: str):
#         prompt = f"""
#         [SYSTEM]
#         You are a professional designer tasked with creating search terms to find images that fit each section of a website landing page. Based on the provided Section_context, generate specific and relevant search terms.

#         #### INSTRUCTIONS ####
#         1. Review the Section_context to understand its content and purpose.
#         2. Create search terms that are highly specific, capturing the semantic essence of the content and suitable for image searches on a website.
#         3. Focus the search terms on the industry theme (e.g., IT, Health Care, Food, Education) identified in Section_context; infer the industry if not explicitly stated.
#         4. Generate exactly 5 keywords in English that reflect key concepts, entities, or themes from the content.
#         5. From the keywords, select or derive 3 unit search terms in English, each 1 or 2 words long, optimized for concise image searches.
#         6. Ensure all terms are in English, relevant to the industry, and visually descriptive (e.g., objects, actions, or settings).
#         7. Output only a JSON object with two keys: "keywords" (list of 5 strings) and "unit_search_terms" (list of 3 strings), without additional text or metadata.

#         [USER_EXAMPLE]
#         Section_context = "KG이니시스는 결제 서비스와 기술 분야의 선도 기업으로, 통합 간편 결제 솔루션을 제공합니다. 1998년에 설립되어 16만 가맹점을 보유하며, 연간 48억 건의 결제를 처리합니다."

#         [ASSISTANT_EXAMPLE]
#         {{
#             "keywords": ["payment solutions", "technology leader", "KG Inisys", "e-commerce", "digital transactions"],
#             "unit_search_terms": ["payment tech", "e-commerce", "digital payment"]
#         }}

#         [USER]
#         Section_context: {context}
#         """
#         # prompt = f"""
#         # [System]
#         # You are a professional designer who make a search term to search images that will fit in each section of the website landing page.
        
#         # #### INSTRUCTIONS ####
#         # 1. TO CREATE SEARCH TERMS, FIRST REVIEW THE USER SECTION CONTEXT.
#         # 2. THE SEARCH TERMS SHOULD BE SPECIFIC AS POSSIBLE, CAPTURING THE SEMANTIC ESSENCE OF THE CONTENT.
#         # 3. FOCUS SEARCH TERMS FOR SEARCHING IMAGE ON THE WEBSITE, WITH THE SUBJECT BEING INDUSTRY (FOR EXAMPLE, 'IT', 'HEALTH CARE', 'FOOD', 'EDUCATION')
#         # 4. MAKE 5 KEYWORDS IN ENGLISH ABOUT THE CONTENT AS THE SEARCH TERM.
#         # 5. CHOOSE 3 UNIT SEARCH TERMS IN ENGLISH WITH 1 OR 2 WORDS.
#         # 6. ENSURE THAT THE OUTPUT LANGUAGE IS ENGLISH.

#         # [/System]

#         # [User_Example]
#         # [/User_Example]
        
#         # [Assistant_Example]
#         # keywords = ['example1', 'example2', 'example3']
#         # [/Assistant_Example]
        
        

#         # [User]
#         # section context = {context}
#         # [/User]
        
#         # """
#         result = await self.send_request(prompt)
#         result.data.generations[0][0].text = self.extract_list(result.data.generations[0][0].text)
#         if result.success:
#             response = result
#             print(f"Section structure response: {response}")
#             return response
#         else:
#             print(f"Section structure generation error: {result.error}")
#             return ""

#     async def section_keyword_create_logic(self, context: str):
#         try:
#             repeat_count = 0
#             while repeat_count < 3:
#                 try:
#                     section_context = await self.section_keyword_recommend(context)
#                     # section_data_with_keyword = await self.process_menu_data(section_context.data.generations[0][0].text.strip())
#                     # section_data_with_keyword = await self.process_data(section_data_with_keyword.data.generations[0][0].text.strip())
#                     return section_context
#                 except RuntimeError as r:
#                     print(f"Runtime error: {r}")
#                     repeat_count += 1
#                 except Exception as e:
#                     print(f"Unexpected error: {e}")
#                     repeat_count += 1
#         except Exception as e:
#             print(f"Error processing landing structure: {e}")
#             return "error"

#     def clean_keyword(self, keyword):
#         cleaned = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', keyword)
#         cleaned = re.sub(r'-', ' ', cleaned)
#         cleaned = re.sub(r'_', ' ', cleaned)
#         cleaned = re.sub(r'[^\w\s]', '', cleaned)
#         return cleaned

#     async def process_data(self, section_context):
#         if isinstance(section_context, dict):
#             section_context = {
#                 key: self.clean_keyword(value) if isinstance(value, str) else value
#                 for key, value in section_context.items()
#             }
#         elif isinstance(section_context, list):
#             section_context = [
#                 self.clean_keyword(item) if isinstance(item, str) else item
#                 for item in section_context
#             ]
#         print(section_context, "<====keyword")
#         return section_context

#     def extract_list(self, text):
#         # 줄바꿈, 캐리지 리턴, 백슬래시 제거
#         text = re.sub(r'[\n\r\\]', '', text)
        
#         # 대괄호를 포함한 전체 리스트를 찾습니다.
#         list_match = re.search(r'\[.*?\]', text, re.DOTALL)
#         if list_match:
#             return list_match.group(0).strip()
#         else:
#             # 대괄호가 없는 경우, 콤마로 구분된 항목들을 대괄호로 감싸줍니다.
#             items = re.findall(r"'(.*?)'", text)
#             if items:
#                 return f"""[{", ".join(f"'{item}'" for item in items)}]"""
#             return None

import json
import re
import asyncio
from src.utils.batch_handler import BatchRequestHandler

class OpenAIKeywordClient:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler

    async def send_request(self, prompt: str, max_tokens: int = 100) -> 'RequestResult':
        try:
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                    "prompt": prompt,
                    "max_tokens": max_tokens,  # 키워드 3개에 적합한 짧은 출력
                    "temperature": 0.7,  # 자연스러운 생성
                    "top_p": 0.9,       # 다양성 확보
                    "n": 1,
                    "stream": False,
                    "logprobs": None
                }, request_id=0),
                timeout=120
            )
            return response
        except asyncio.TimeoutError:
            print("[ERROR] Request timed out")
            return "Error: Request timed out after 60 seconds"
        except Exception as e:
            print(f"[ERROR] Unexpected error: {str(e)}")
            return f"Error: {str(e)}"

    def clean_keyword(self, keyword):
        cleaned = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', keyword)
        cleaned = re.sub(r'[-_]', ' ', cleaned)
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        return cleaned.strip()

    def extract_keywords(self, result):
        default_keywords = ["industry concept", "generic term", "basic idea"]

        if result.success and hasattr(result, 'data') and result.data.generations:
            text = result.data.generations[0][0].text
            try:
                # JSON 문자열에서 실제 JSON 객체 추출
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    
                    if isinstance(json_data, dict) and "keyword" in json_data:
                        keywords = json_data["keyword"]
                    elif isinstance(json_data, list):
                        keywords = json_data
                    else:
                        keywords = []
                        
                    # 키워드 정제 및 중복 제거
                    keywords = list(set([self.clean_keyword(kw) for kw in keywords if isinstance(kw, str)]))
                    
                    # 정확히 3개의 키워드 확보
                    keywords = keywords[:3]
                    keywords += default_keywords[len(keywords):3]
                    
                    # 키워드가 비어있는지 확인
                    if not keywords or len(keywords) < 3:
                        print("[WARNING] Extracted keywords are less than 3, using defaults")
                        keywords = default_keywords
                else:
                    print("[WARNING] No JSON match found in text")
                    keywords = default_keywords
            except json.JSONDecodeError:
                print("[ERROR] Failed to parse JSON")
                keywords = default_keywords
        else:
            keywords = default_keywords

        # 리스트 자체를 반환하도록 수정
        return keywords


    # def extract_keywords(self, result):
    #     """Extract and clean exactly 3 keywords, wrap in 'keyword' object"""
    #     print(f"[DEBUG] Extracting keywords from result: {result}")
    #     if result.success and hasattr(result, 'data') and result.data.generations:
    #         text = result.data.generations[0][0].text
    #         try:
    #             json_data = json.loads(text)
    #             if isinstance(json_data, dict) and "keyword" in json_data:
    #                 keywords = [self.clean_keyword(kw) for kw in json_data["keyword"][:3]]
    #             elif isinstance(json_data, list):
    #                 keywords = [self.clean_keyword(kw) for kw in json_data[:3]]
    #             else:
    #                 keywords = ["parsing error", "invalid format", "try again"]
    #             return json.dumps({"keyword": keywords}, ensure_ascii=False)
    #         except json.JSONDecodeError:
    #             cleaned_text = self.clean_text(text)
    #             try:
    #                 json_data = json.loads(cleaned_text)
    #                 keywords = [self.clean_keyword(kw) for kw in json_data["keyword"][:3]] if "keyword" in json_data else [self.clean_keyword(kw) for kw in json_data[:3]]
    #                 return json.dumps({"keyword": keywords}, ensure_ascii=False)
    #             except json.JSONDecodeError:
    #                 # JSON 파싱 실패 시 기본 키워드 제공
    #                 return json.dumps({"keyword": ["industry concept", "generic term", "basic idea"]}, ensure_ascii=False)
    #     return json.dumps({"keyword": ["extraction failed", "no data", "error"]}, ensure_ascii=False)

    def clean_text(self, text):
        """Remove unwanted tags and clean text"""
        text = re.sub(r'[\n\r\\\\/]', '', text, flags=re.DOTALL)
        headers_to_remove = [
            "<|start_header_id|>system<|end_header_id|>",
            "<|start_header_id|>SYSTEM<|end_header_id|>",
            "<|start_header_id|>", "<|end_header_id|>",
            "<|start_header_id|>user<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|eot_id|>", "[SYSTEM]", "[USER]", "ASSISTANT",
            "[ASSISTANT_SOLUTION]"
        ]
        for header in headers_to_remove:
            text = text.replace(header, '')
        text = re.sub(r'<\|.*?\|>', '', text)
        return text.strip()

    async def section_keyword_recommend(self, context: str, max_tokens: int = 100) -> 'RequestResult':
        # prompt = f"""
        # [SYSTEM]
        # You are a professional designer tasked with creating search terms to find images that fit each section of a website landing page. Based on the provided Section_context, generate specific and relevant search terms.

        # #### INSTRUCTIONS ####
        # 1. Review the Section_context to understand its content and purpose.
        # 2. Create search terms that are highly specific, capturing the semantic essence of the content and suitable for image searches on a website.
        # 3. Focus the search terms on the industry theme (e.g., IT, Health Care, Food, Education) identified in Section_context; infer the industry if not explicitly stated.
        # 4. Generate exactly 3 keywords in English that reflect key concepts, entities, or themes from the content.
        # 5. Ensure all terms are in English, relevant to the industry, and visually descriptive (e.g., objects, actions, or settings).
        # 6. If Section_context is empty, use generic terms like "industry concept" while maintaining the structure.
        # 7. Output only a JSON object with one key: "keyword" (list of exactly 3 strings), without additional text, metadata, or other keys.

        # [USER_EXAMPLE]
        # Section_context = "KG이니시스는 결제 서비스와 기술 분야의 선도 기업으로, 통합 간편 결제 솔루션을 제공합니다. 1998년에 설립되어 16만 가맹점을 보유하며, 연간 48억 건의 결제를 처리합니다."

        # [ASSISTANT_EXAMPLE]
        # {{"keyword": ["payment solutions", "KG Inisys", "e-commerce"]}}

        # [USER]
        # Section_context: {context}
        # """
        
        sys_prompt = f"""
            You are a professional designer tasked with creating search terms to find images that fit each section of a website landing page. Based on the provided Section_context, generate specific and relevant search terms.

            #### INSTRUCTIONS ####
            1. Review the Section_context to understand its content and purpose.
            2. Create search terms that are highly specific, capturing the semantic essence of the content and suitable for image searches on a website.
            3. Focus the search terms on the industry theme (e.g., IT, Health Care, Food, Education) identified in Section_context; infer the industry if not explicitly stated.
            4. Generate exactly 3 keywords in English that reflect key concepts, entities, or themes from the content.
            5. Ensure all terms are in English, relevant to the industry, and visually descriptive (e.g., objects, actions, or settings).
            6. If Section_context is empty, use generic terms like "industry concept" while maintaining the structure.
            7. Output only a JSON object with one key: "keyword" (list of exactly 3 strings), without additional text, metadata, or other keys.

            [USER_EXAMPLE]
            Section_context = "KG이니시스는 결제 서비스와 기술 분야의 선도 기업으로, 통합 간편 결제 솔루션을 제공합니다. 1998년에 설립되어 16만 가맹점을 보유하며, 연간 48억 건의 결제를 처리합니다."

            [ASSISTANT_EXAMPLE]
            {{"keyword": ["payment solutions", "KG Inisys", "e-commerce"]}}
        """
        
        usr_prompt = f"""
            Section_context: {context}
        """        
        max_attempts = 3
        last_result = None
        
        extra_body = {
            "guided_json": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "array",
                        "items": {
                            "type": "string",  # 각 항목은 문자열
                            "pattern": "^[a-zA-Z0-9\\s-]*$" # 영어 문자, 숫자, 공백, 하이픈만 허용
                        },
                        "minItems": 3,  # 최소 3개
                        "maxItems": 3   # 최대 3개
                    }
                },
                "required": ["keyword"]  # keyword 필수
            }
        }
        
        result = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "sys_prompt": sys_prompt,
                "usr_prompt": usr_prompt,
                "extra_body": extra_body,
                "max_tokens": max_tokens,  # 키워드 3개에 적합한 짧은 출력
                "temperature": 0.7,  # 자연스러운 생성
                "top_p": 0.9,       # 다양성 확보
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=120
        )
        
        temp_result = result.data['generations'][0][0]['text']
        
        json_temp_result = json.loads(temp_result)
        
        only_keywords = json_temp_result["keyword"]
        print("only_keywords : ", only_keywords)
        
        result.data['generations'][0][0]['text'] = only_keywords
        
        return result
        
        # print("type(keywords) :", type(result.data['generations'][0][0]['text']))
        # print("keywords :", result.data['generations'][0][0]['text'])
        # if result.success and hasattr(result, 'data'):
            
            # print("keywords :", result.data.generations[0][0].text)
            
            # return result
        
        # for attempt in range(max_attempts):
        #     result = await asyncio.wait_for(
        #         self.batch_handler.process_single_request({
        #             "prompts": prompt
        #             "max_tokens": max_tokens,  # 키워드 3개에 적합한 짧은 출력
        #             "temperature": 0.7,  # 자연스러운 생성
        #             "top_p": 0.9,       # 다양성 확보
        #             "n": 1,
        #             "stream": False,
        #             "logprobs": None
        #         }, request_id=0),
        #         timeout=120
        #     )
        #     last_result = result  # 마지막 결과 저장
            
        #     if isinstance(result, str):  # 에러 메시지 처리
        #         print(f"[ERROR] Request returned error: {result}")
        #         return result
            
        #     if result.success and hasattr(result, 'data'):
        #         keywords = self.extract_keywords(result)
                
        #         # 키워드가 비어있거나 3개 미만인 경우 재시도
        #         if not keywords or len(keywords) < 3:
        #             print(f"[WARNING] Generated keywords are empty or less than 3. Attempt {attempt + 1}/{max_attempts}")
        #             continue
                    
        #         # 한글이 포함된 키워드가 있는지 확인
        #         contains_korean = any(any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in keyword) for keyword in keywords)
                
        #         if not contains_korean:
        #             result.data.generations[0][0].text = keywords
        #             return result
        #         else:
        #             print(f"[WARNING] Generated keywords contain Korean. Attempt {attempt + 1}/{max_attempts}")
        
        # 모든 시도 실패 시 기본 키워드로 덮어씌우기
        if last_result and last_result.success and hasattr(last_result, 'data'):
            default_keywords = ["business concept", "professional service", "corporate solution"]
            last_result.data.generations[0][0].text = default_keywords
            return last_result
        else:
            # 마지막 결과가 없거나 유효하지 않은 경우 에러 반환
            return "Failed to generate valid keywords"

    async def section_keyword_create_logic(self, context: str, max_tokens: int = 100) -> 'RequestResult':
        try:
            repeat_count = 0
            while repeat_count < 3:
                try:
                    result = await self.section_keyword_recommend(context, max_tokens)
                    if result.success:
                        return result
                    print(f"[WARN] Attempt {repeat_count + 1} failed: {result.error}")
                    repeat_count += 1
                except RuntimeError as r:
                    print(f"[WARN] Runtime error (attempt {repeat_count + 1}): {r}")
                    repeat_count += 1
                except Exception as e:
                    print(f"[WARN] Unexpected error (attempt {repeat_count + 1}): {e}")
                    repeat_count += 1
            print("[ERROR] All retry attempts failed")
            return "Error: All retry attempts failed"
        except Exception as e:
            print(f"[ERROR] Fatal error processing keywords: {e}")
            return f"Error: {str(e)}"