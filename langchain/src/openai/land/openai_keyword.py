import json
import re
import asyncio
from src.utils.batch_handler import BatchRequestHandler


class OpenAIKeywordClient:
    def __init__(self, batch_handler: BatchRequestHandler):
        self.batch_handler = batch_handler

    async def send_request(self, prompt: str) -> str:
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.1,
                "top_p": 1.0,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=60  # 적절한 타임아웃 값 설정
        )
        return response
    
    async def process_menu_data(self, menu_data: str) -> list:
        try:
            print(f"menu_data : {menu_data}")
            json_match = re.search(r"\[.*\]", menu_data, re.DOTALL)
            if not json_match:
                raise ValueError("JSON 형식을 찾을 수 없습니다.")
            json_text = json_match.group()
            json_text = json_text.replace("'", '"').replace("\n", "").strip()
            json_data = json.loads(json_text)
            return json_data
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {e}")
            raise RuntimeError("menu_data의 형식이 올바르지 않습니다.") from e

    async def section_keyword_recommend(self, context: str):
        prompt = f"""
        System:
        You are a professional designer who make a search term to search images that will fit in each section of the website landing page.
        
        #### Instructions ####
        1. To create search terms, first review the User section context.
        2. The search terms should be specific as possible, capturing the semantic essence of the content.
        3. Focus search terms for searching image on the website, with the subject being industry (for example, 'IT', 'health care', 'food', 'education')
        4. Make 5 keywords in English about the content as the search term.
        5. Choose 3 unit search terms in English with 1 or 2 words.
        6. ensure that the output language is English.
        
        #### Example Output ####
        keywords = ['example1', 'example2', 'example3']
        
        User:
        section context = {context}
        """
        result = await self.send_request(prompt)
        if result.success:
            response = result
            print(f"Section structure response: {response}")
            return response
        else:
            print(f"Section structure generation error: {result.error}")
            return ""

    async def section_keyword_create_logic(self, context: str):
        try:
            print(context,"<=====context")
            repeat_count = 0
            while repeat_count < 3:
                try:
                    section_context = await self.section_keyword_recommend(context)
                    # section_data_with_keyword = await self.process_menu_data(section_context.data.generations[0][0].text.strip())
                    # section_data_with_keyword = await self.process_data(section_data_with_keyword.data.generations[0][0].text.strip())
                    return section_context
                except RuntimeError as r:
                    print(f"Runtime error: {r}")
                    repeat_count += 1
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    repeat_count += 1
        except Exception as e:
            print(f"Error processing landing structure: {e}")
            return "error"

    def clean_keyword(self, keyword):
        cleaned = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', keyword)
        cleaned = re.sub(r'-', ' ', cleaned)
        cleaned = re.sub(r'_', ' ', cleaned)
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        return cleaned

    async def process_data(self, section_context):
        if isinstance(section_context, dict):
            section_context = {
                key: self.clean_keyword(value) if isinstance(value, str) else value
                for key, value in section_context.items()
            }
        elif isinstance(section_context, list):
            section_context = [
                self.clean_keyword(item) if isinstance(item, str) else item
                for item in section_context
            ]
        print(section_context, "<====keyword")
        return section_context
