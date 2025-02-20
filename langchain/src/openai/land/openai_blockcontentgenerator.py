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
                "temperature": 0.6,
                "top_p": 0.4,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=120  # 적절한 타임아웃 값 설정
        )
        return response

    async def generate_content(self, tag_length: Dict[str, Any], section_context: Dict[str, Any]) -> Dict[str, Any]:
        # NOTE 250220 : F.E에서 받은거 형식 넣기
    
        
        # html = self.emmet_parser.parse_emmet(next(iter(select_block.values())))
        # print(f"html : {html}")
        # print(f"next(iter(section_context.keys())) : {next(iter(section_context.keys()))}")
        # style = self.emmet_parser.font_size(next(iter(section_context.keys())))
        # print(f"style : {style}")

        prompt = f"""
        [System]
        You are an AI assistant that uses Section_context to create content for each semantic tag.
        If the user provides Section_context and json_type_tag_list, read the content first and then create content to enter the semantic tag in each key.

        #### Explain Semantic Tag ####
        h1: Represents the most important title of the web page. Typically only one is used per page, and it represents the topic or purpose of the page.
        h2: Indicates the next most important heading after h1. It is mainly used to separate major sections of a page.
        h3: A subheading of h2, indicating detailed topics within the section.
        h5: It is located around h tags and briefly assists them.
        p: A tag that defines a paragraph. Used to group plain text content.
        li: Represents a list item. Be sure that the inner tags should have similar shape.

        #### Instructions ####

        1. DO NOT CHANGE THE JSON STRUCTURE PROVIDED BY THE USER.
        2. DO NOT INSERT ADDITIONAL TEXT OTHER THAN THE VALUE OF EACH KEY.
        3. ENSURE THAT THE OUTPUT MATCHES THE JSON OUTPUT EXAMPLE BELOW.

        [/System]
        
        [User_Example]
        Section_context = "Section context"
        json_type_tag_list = 
        {{
            "h1": "int type text length",
            "h2": "int type text length",
            "p": "int type text length",
            "li": [
                {{"h2": "int type text length", 
                "p": "int type text length"}},
                {{"h2": "int type text length", 
                "p": "int type text length"}}
            ]
        }}
        [/User_Example]
        
        [Assistant_Example]
        Output : 
        {{
            "h1": "Description",
            "h2": "Description",
            "p": "Description",
            "li": [
                {{"h2": "Description", 
                "p": "Description"}},
                {{"h2": "Description", 
                "p": "Description"}}
            ]
        }}
        [/Assistant_Example]
        

        [User]
        Section_context: {section_context}
        json_type_tag_list : {tag_length}
        [/User]
        
        """

        result = await self.send_request(prompt)

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