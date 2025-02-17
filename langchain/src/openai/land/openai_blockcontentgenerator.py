import asyncio
from typing import Dict, Any
import re
import json
from src.utils.emmet_parser import EmmetParser


class OpenAIBlockContentGenerator:
    def __init__(self, batch_handler):
        self.batch_handler = batch_handler
        self.emmet_parser = EmmetParser()

    async def send_request(self, prompt: str) -> str:
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 1.0,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=120  # 적절한 타임아웃 값 설정
        )
        return response

    async def generate_content(self, select_block: Dict[str, Any], section_context: Dict[str, Any]) -> Dict[str, Any]:
        html = self.emmet_parser.parse_emmet(next(iter(select_block.values())))
        print(f"html : {html}")
        print(f"next(iter(section_context.keys())) : {next(iter(section_context.keys()))}")
        style = self.emmet_parser.font_size(next(iter(section_context.keys())))
        print(f"style : {style}")

        prompt = f"""
        System:
        You are an AI assistant that uses Section_context to create content for each semantic tag.
        If the user provides Section_context and json_type_input, read the content first and then create content to enter the semantic tag in each key.

        #### Explain Semantic Tag ####
        h1: Represents the most important title of the web page. Typically only one is used per page, and it represents the topic or purpose of the page.
        h2: Indicates the next most important heading after h1. It is mainly used to separate major sections of a page.
        h3: A subheading of h2, indicating detailed topics within the section.
        subtitle: It is located around h tags and briefly assists them.
        p: A tag that defines a paragraph. Used to group plain text content.
        li: Represents a list item. Mainly used within ul (unordered list) or ol (ordered list) tags.

        #### Instructions ####

        1. Do not change the json structure provided by the user.
        2. Do not insert additional text other than the value of each key.
        3. ensure that the output matches the JSON output example below.
        4. Follow these style guidelines:
        {style}

        #### Example JSON Output ####
        ```
        {{
            "h1": "description",
            "h2": "description",
            "p": "Description",
            "li": [
                {{"h2": "description", "p": "Description"}},
                {{"h2": "description", "p": "Description"}}
            ]
        }}
        ```

        User:
        Section_context: {section_context}
        """

        result = await self.send_request(prompt)
        gen_content = self.emmet_parser.tag_sort(result.data.generations[0][0].text.strip())
        gen_content = self.extract_json(gen_content)
        result.data.generations[0][0].text = gen_content
        # if not self.emmet_parser.validate_html_structure(gen_content, html):
        #     raise ValueError(f"생성된 HTML 구조가 예상과 다릅니다: {gen_content}")
        return {
            'HTML_Tag': next(iter(select_block.values())),
            'Block_id': next(iter(select_block.keys())),
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