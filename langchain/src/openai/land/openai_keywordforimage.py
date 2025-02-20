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
        [System]
        You are a professional designer who make a search term to search images that will fit in each section of the website landing page.
        
        #### INSTRUCTIONS ####
        1. TO CREATE SEARCH TERMS, FIRST REVIEW THE USER SECTION CONTEXT.
        2. THE SEARCH TERMS SHOULD BE SPECIFIC AS POSSIBLE, CAPTURING THE SEMANTIC ESSENCE OF THE CONTENT.
        3. FOCUS SEARCH TERMS FOR SEARCHING IMAGE ON THE WEBSITE, WITH THE SUBJECT BEING INDUSTRY (FOR EXAMPLE, 'IT', 'HEALTH CARE', 'FOOD', 'EDUCATION')
        4. MAKE 5 KEYWORDS IN ENGLISH ABOUT THE CONTENT AS THE SEARCH TERM.
        5. CHOOSE 3 UNIT SEARCH TERMS IN ENGLISH WITH 1 OR 2 WORDS.
        6. ENSURE THAT THE OUTPUT LANGUAGE IS ENGLISH.

        [/System]

        [User_Example]
        [/User_Example]
        
        [Assistant_Example]
        keywords = ['example1', 'example2', 'example3']
        [/Assistant_Example]
        
        

        [User]
        section context = {context}
        [/User]
        
        """
        result = await self.send_request(prompt)
        result.data.generations[0][0].text = self.extract_list(result.data.generations[0][0].text)
        if result.success:
            response = result
            print(f"Section structure response: {response}")
            return response
        else:
            print(f"Section structure generation error: {result.error}")
            return ""

    async def section_keyword_create_logic(self, context: str):
        try:
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

    def extract_list(self, text):
        # 줄바꿈, 캐리지 리턴, 백슬래시 제거
        text = re.sub(r'[\n\r\\]', '', text)
        
        # 대괄호를 포함한 전체 리스트를 찾습니다.
        list_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if list_match:
            return list_match.group(0).strip()
        else:
            # 대괄호가 없는 경우, 콤마로 구분된 항목들을 대괄호로 감싸줍니다.
            items = re.findall(r"'(.*?)'", text)
            if items:
                return f"""[{", ".join(f"'{item}'" for item in items)}]"""
            return None