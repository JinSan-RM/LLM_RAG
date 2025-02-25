import asyncio
from typing import Dict, Any
import re
import json
from src.utils.emmet_parser import EmmetParser


class OpenAIBlockContentGenerator:
    def __init__(self, batch_handler):
        self.batch_handler = batch_handler
        self.emmet_parser = EmmetParser()

    async def send_request(self, prompt: str, max_tokens: int = 1000) -> str:
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.6,
                "top_p": 0.4,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=120  # 적절한 타임아웃 값 설정
        )
        return response

    async def generate_content(
        self, 
        tag_length: Dict[str, Any], 
        section_context: Dict[str, Any], 
        max_tokens: int = 1000) -> Dict[str, Any]:
        # NOTE 250220 : F.E에서 받은거 형식 넣기
    
        
        # html = self.emmet_parser.parse_emmet(next(iter(select_block.values())))
        # print(f"html : {html}")
        # print(f"next(iter(section_context.keys())) : {next(iter(section_context.keys()))}")
        # style = self.emmet_parser.font_size(next(iter(section_context.keys())))
        # print(f"style : {style}")

        prompt = f"""
        [SYSTEM]
        You are an AI assistant that generates content for semantic tags based on the provided Section_context. 
        When given Section_context and json_type_tag_list, read the Section_context first, then create content for each semantic tag key in the json_type_tag_list and the value is max tokens for each content, ensuring the content reflects the context.

        #### Semantic Tag Definitions ####
        - h1: The primary title of the web page, summarizing its main topic or purpose (typically one per page).
        - h2: Major section headings, separating key parts of the page.
        - h3: Subheadings under h2, detailing specific topics within sections.
        - h5: Brief supporting text around h tags, enhancing their meaning.
        - p: Paragraphs of plain text, grouping descriptive content.
        - li: List items, where inner tags (e.g., h2, p) should maintain a consistent structure.

        #### Instructions ####
        1. Read the Section_context and use it as the basis for generating content for each tag in json_type_tag_list.
        2. Match the output language to the primary language of Section_context (e.g., Korean or English).
        3. For each key in json_type_tag_list, replace the value with content derived from Section_context, tailored to the tag's purpose (e.g., h1 for main title, p for detailed text).
        4. If json_type_tag_list values are integers, interpret them as desired character lengths and generate content approximately matching that length.
        5. Preserve the exact JSON structure provided in json_type_tag_list, only replacing values with generated content.
        6. Do not add extra text or keys beyond the provided structure.
        7. Ensure content is concise, relevant, and avoids repetition across tags.
        8. Output only the resulting JSON, without additional tags like [SYSTEM] or metadata.

        [USER_EXAMPLE]
        Section_context = "KG이니시스는 결제 서비스와 기술 분야의 선도 기업으로, 통합 간편 결제 솔루션을 제공합니다. 1998년에 설립되어 16만 가맹점을 보유하며, 연간 48억 건의 결제를 처리합니다."
        json_type_tag_list = 
        {{
            "h1": "50",
            "h2": "100",
            "p": "200",
            "li": [
                {{"h2": "50", "p": "150"}},
                {{"h2": "50", "p": "150"}}
            ]
        }}

        [ASSISTANT_EXAMPLE]
        {{
            "h1": "KG이니시스: 결제 서비스의 리더",
            "h2": "통합 간편 결제 솔루션으로 시장을 선도하는 KG이니시스",
            "p": "KG이니시스는 1998년 설립된 결제 기술 분야의 선도 기업으로, 16만 가맹점을 통해 연간 48억 건의 결제를 처리하며, 고객에게 편리하고 신뢰할 수 있는 통합 결제 솔루션을 제공합니다.",
            "li": [
                {{"h2": "가맹점 네트워크의 강점", "p": "16만 개 이상의 가맹점을 보유한 KG이니시스는 결제 서비스의 안정성과 접근성을 높이며, 다양한 사업者に 편리한 결제 경험을 제공합니다."}},
                {{"h2": "대규모 결제 처리 역량", "p": "연간 48억 건의 결제를 처리하는 KG이니시스는 기술력을 바탕으로 빠르고 안전한 결제 솔루션을 통해 시장에서 독보적인 위치를 유지합니다."}}
            ]
        }}

        [USER]
        Section_context: {section_context}
        json_type_tag_list: {tag_length}
        """

        result = await self.send_request(prompt, max_tokens)

        gen_content = self.emmet_parser.tag_sort(result.data.generations[0][0].text.strip())
        gen_content = self.extract_json(gen_content)
        result.data.generations[0][0].text = gen_content
        # if not self.emmet_parser.validate_html_structure(gen_content, html):
        #     raise ValueError(f"생성된 HTML 구조가 예상과 다릅니다: {gen_content}")
        
        return {
            # NOTE 250220 : "/api/block_select" 결과로 미리 보내므로 필요 없어짐짐
            # 'HTML_Tag': next(iter(select_block.values())),
            # 'Block_id': next(iter(select_block.keys())),
            'gen_content': result
        }

    def extract_json(self, text):
        # 가장 바깥쪽의 중괄호 쌍을 찾습니다.
        text = re.sub(r'[\n\r\\\\/]', '', text, flags=re.DOTALL)
        json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            # Handle case where only opening brace is found
            json_str = re.search(r'\{.*', text, re.DOTALL)
            if json_str:
                json_str = json_str.group() + '}'
            else:
                return None

        # Balance braces if necessary
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try a more lenient parsing approach
            try:
                return json.loads(json_str.replace("'", '"'))
            except json.JSONDecodeError:
                return None