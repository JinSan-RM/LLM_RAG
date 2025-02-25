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
    
        print("Want to see how it looks like tag_length : ", tag_length)
        # html = self.emmet_parser.parse_emmet(next(iter(select_block.values())))
        # print(f"html : {html}")
        # print(f"next(iter(section_context.keys())) : {next(iter(section_context.keys()))}")
        # style = self.emmet_parser.font_size(next(iter(section_context.keys())))
        # print(f"style : {style}")

        prompt = f"""
        [SYSTEM]
        You are an AI assistant that generates content for semantic tags based on the provided Section_context. 
        When given Section_context and json_type_tag_list, read the Section_context first, then create content for each semantic tag key in the json_type_tag_list and replace its value with generated content. 
        Ensure each tag's content aligns with its purpose while strictly adhering to the maximum character length specified in json_type_tag_list.

        #### Semantic Tag Definitions ####
        - h1: The primary title of the web page, summarizing its main topic or purpose (typically one per page).
        - h2: Major section headings, separating key parts of the page.
        - h3: Subheadings under h2, detailing specific topics within sections.
        - h5: Brief supporting text around h tags, enhancing their meaning.
        - p: Paragraphs of plain text, grouping descriptive content.
        - li: List items, where inner tags (e.g., h2, p) should maintain a consistent structure.

        #### Instructions ####
        1. **READ THE SECTION_CONTEXT AND USE IT AS THE BASIS FOR GENERATING CONTENT FOR EACH TAG IN JSON_TYPE_TAG_LIST.**
        2. **MATCH THE OUTPUT LANGUAGE TO THE PRIMARY LANGUAGE OF SECTION_CONTEXT (E.G., KOREAN OR ENGLISH).**
        3. **FOR EACH KEY IN JSON_TYPE_TAG_LIST, THE VALUE REPRESENTS THE MAXIMUM CHARACTER LENGTH.**
        - If the value is an integer, generate text that does NOT exceed this length.
        - If text needs to be shortened, keep the core meaning while reducing words.
        - Ensure readability and maintain natural sentence structure.
        4. **PRESERVE THE EXACT JSON STRUCTURE PROVIDED IN JSON_TYPE_TAG_LIST, ONLY REPLACING VALUES WITH GENERATED CONTENT.**
        5. **ENSURE CONTENT IS CONCISE, RELEVANT, AND AVOIDS REPETITION ACROSS TAGS.**
        6. **OUTPUT ONLY THE RESULTING JSON, WITHOUT ADDITIONAL TAGS LIKE [SYSTEM] OR METADATA.**
        7. **IF A LIST STRUCTURE EXISTS IN JSON_TYPE_TAG_LIST, GENERATE MULTIPLE ENTRIES WHILE ENSURING EACH MAINTAINS THE DESIGNATED MAXIMUM CHARACTER LENGTH.**

        [USER_EXAMPLE]
        Section_context = "재밋은 AI 솔루션을 기반으로 사용자들에게 간단하고 편리하게 웹 사이트를 만들 수 있도록 도와주는 선도 서비스입니다. 기업 '위븐'은 AI 솔루션을 통해 일반인들도 쉽게 접근할 수 있으며, 전문가가 사용해도 무방한 에디터와 스튜디오 서비스를 보유하고 있어서 다방면에 능한 서비스를 갖고 있는 기업입니다."
        json_type_tag_list = 
        {{
            "h1_0": "15",
            "h2_0": "15",
            "p_0": "30",
            "li_0": [
                {{"h2_0": "15", "p_0": "40"}},
                {{"h2_1": "15", "p_0": "40"}}
            ],
            "p_1: "70"
        }}

        [ASSISTANT_EXAMPLE]
        {{
            "h1_0": "AI로 간편하게 만드는 웹사이트",
            "h2_0": "누구나 쉽게 활용하는 AI 웹 제작",
            "p_0": "위븐은 다방면에 능한 AI 웹 제작 서비스를 제공합니다.",
            "li_0": [
                {{
                    "h2_0": "AI 웹 제작 혁신",
                    "p_0": "기업 '위븐'의 AI 솔루션은 누구나 직관적으로 웹사이트를 만들 수 있도록 지원합니다."
                }},
                {{
                    "h2_1": "전문가도 만족하는 기능",
                    "p_1": "초보자는 물론 전문가도 활용 가능한 강력한 에디터와 스튜디오 기능을 제공합니다."
                }}
            ],
            "p_1": "재밋은 AI 기반 웹사이트 제작 솔루션을 제공하는 선도 서비스로, 일반 사용자부터 전문가까지 쉽게 활용할 수 있는 강력한 에디터와 스튜디오 서비스를 갖추고 있습니다."
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