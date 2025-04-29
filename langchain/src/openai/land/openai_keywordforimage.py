import json
import re
import asyncio
from src.utils.batch_handler import BatchRequestHandler

class OpenAIKeywordClient:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler

    async def send_request(self, sys_prompt: str, usr_prompt: str, extra_body, max_tokens: int = 100) -> str:
        model = "/usr/local/bin/models/gemma-3-4b-it"
        try:
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                    "model": model,
                    "sys_prompt": sys_prompt,
                    "usr_prompt": usr_prompt,
                    "extra_body": extra_body,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,  # 안정성 우선
                    "top_p": 0.9,
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

    async def section_keyword_recommend(self, context: str, max_tokens: int = 100) -> 'RequestResult':
        
        sys_prompt = f"""
            You are a professional designer tasked with creating search terms to find images that fit a website landing page. 
            Based on the provided {{Section_context}}, generate universal and relevant search terms.

            #### INSTRUCTIONS ####
            1. Review the Section_context to understand its content and purpose.
            2. Create search terms that are highly universal, capturing the semantic essence of the content and suitable for image searches on a website.
            3. Focus the search terms on the industry theme (e.g., IT, Health Care, Food, Education) identified in Section_context. Infer the industry if not explicitly stated.
            4. Generate exactly 3 keywords in English that reflect "industry, key concepts, themes" from the Section_context.
            5. Ensure all terms are in English, relevant to the industry, and visually descriptive for using by image searching terms.
            6. If Section_context is empty, use generic terms like "industry concept" while maintaining the structure.
            7. Output is follow the extra_body.

            [Section_context_EXAMPLE]
            Section_context = "KG이니시스는 결제 서비스와 기술 분야의 선도 기업으로, 통합 간편 결제 솔루션을 제공합니다. 1998년에 설립되어 16만 가맹점을 보유하며, 연간 48억 건의 결제를 처리합니다."

            [ASSISTANT_EXAMPLE]
            {{"keyword": ["payment solutions", "convenient", "e-commerce"]}}
        """
        
        usr_prompt = f"""
            Section_context: {context}
        """        
        
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

        result = await self.send_request(
            sys_prompt=sys_prompt, 
            usr_prompt=usr_prompt, 
            max_tokens=max_tokens, 
            extra_body=extra_body
            )

        temp_result = result.data['generations'][0][0]['text']

        json_temp_result = json.loads(temp_result)

        only_keywords = json_temp_result["keyword"]

        result.data['generations'][0][0]['text'] = only_keywords

        return result

    async def section_keyword_create_logic(self, context: str, max_tokens: int = 100) -> str:
        try:
            repeat_count = 0
            while repeat_count < 3:
                try:
                    result = await self.section_keyword_recommend(context, max_tokens)
                    if result.success:
                        return result
                    repeat_count += 1
                except RuntimeError as r:
                    print(f"[WARN] Runtime error (attempt {repeat_count + 1}): {r}")
                    repeat_count += 1
                except Exception as e:
                    print(f"[WARN] Unexpected error (attempt {repeat_count + 1}): {e}")
                    repeat_count += 1
            return "Error: All retry attempts failed"
        except Exception as e:
            print(f"[ERROR] Fatal error processing keywords: {e}")
            return f"Error: {str(e)}"