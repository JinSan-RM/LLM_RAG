import asyncio
from typing import List, Dict, Any
import re
import difflib

class OpenAIBlockRecommend:
    def __init__(self, batch_handler):
        self.batch_handler = batch_handler

    async def send_request(self, prompt: str) -> str:
        request = {
            "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
            "prompt": prompt,
            "temperature": 0.1
        }
        response = await self.batch_handler.process_single_request(
                                                    request,
                                                    request_id=0
                                                    )
        if response.success:
            return response.data.generations[0][0].text.strip()
        else:
            raise RuntimeError(f"API 요청 실패: {response.error}")

    async def generate_block_content_batch(
        self, 
        block_lists: List[Dict[str, Any]],
        contexts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        
        tasks = [
            self.generate_block_content(block_list, context)
            for block_list, context in zip(block_lists, contexts)
        ]
        return await asyncio.gather(*tasks)

    async def generate_block_content(
        self, 
        block_list: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        result_dict = {}
        
        for section_name, data_list in block_list.items():
            if section_name in ['Header', 'Footer']:
                result_dict[section_name] = self.process_header_footer(data_list, context.get(section_name, ""))
            else:
                result_dict[section_name] = await self.process_section(section_name, data_list, context.get(section_name, ""), block_list.keys())
        
        return result_dict

    def process_header_footer(
        self,
        data_list: Dict[str, str],
        ctx_value: str
        ) -> Dict[str, Any]:
        
        first_item = next(iter(data_list.items()))
        b_id, b_value = first_item
        b_value = self.extract_emmet_tag(b_value)
        return {
            'HTML_Tag': b_value,
            'Block_id': b_id,
            'gen_content': ctx_value
        }

    async def process_section(self, section_name: str, data_list: Dict[str, str], ctx_value: str, all_sections: List[str]) -> Dict[str, Any]:
        tag_slice = list(data_list.values())
        
        prompt = f"""
            <|start_header_id|>system<|end_header_id|>
            1. {section_name} 섹션에 가장 잘 어울리는 태그 하나를 태그리스트트 중에서 선정하세요.
            2. 단 하나의 태그만 반환하시고, 그 외의 어떤 텍스트도 출력하지 마세요.
            3. 태그의 HTML 구조는 변경하지 말고 그대로 출력하세요.
            4. **주석(<!-- -->), 설명, 빈 줄 등 추가 텍스트나 요약본, 문장(해설, 코드 등)을 삽입하지 마세요.**
            5. **태그리스트 중 하나만** 최종 출력해야 하며, 다른 모든 형식(JSON, 코드 블록 등)은 절대 포함하지 마세요.
            6. 예: {tag_slice}
            7. 만약 태그 리스트 중 하나가 아닌 다른 텍스트가 발견되면, 오류가 발생했다고 판단하고 **재시도**하세요.
            8. 출력예시를 따라 출력하세요.

            <|eot_id|><|start_header_id|>user<|end_header_id|>

            **태그 리스트**:
            {tag_slice}

            출력예시:
            {tag_slice}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            **반드시 태그 리스트 안의 데이터 중 하나만 골라서 그 값만 출력하세요.**
            **최종 출력은 오직 태그만 있어야 하며, 다른 형식(JSON, 코드 블록, 설명 등)이나 텍스트를 절대 포함하면 안 됩니다.**
            """
        raw_json = await self.send_request(prompt)
        
        b_id = self.find_key_by_value(data_list, raw_json)
        if b_id is None:
            print(f"매칭되는 b_id가 없습니다: {section_name}")
            return {}

        b_value = self.extract_emmet_tag(data_list[b_id])
        
        parser = EmmetParser()
        html = parser.parse_emmet(b_value)
        style = parser.font_size(section_name)

        content_prompt = prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        너는 자료를 HTML 태그의 형식에 맞게 정리하는 역할을 할 거야.

        아래 규칙들을 반드시 준수 할 것:
        1. user가 제공한 HTML 태그 구조를 그대로 유지할 것.
        2. 오직 제공된 HTML 태그 내부에만 내용을 작성할 것.
        3. 태그 외의 추가 텍스트(주석, 설명, 빈 줄 등)는 절대 삽입하지 말 것.
        4. 전체 섹션 리스트 중, 현재 섹션을 고려해서 내용을 작성할 것.
        5. **사용자가 입력한 HTML 구조를 반드시 지켜줘.** 자료를 더 넣으려고 태그를 추가하지 말 것.
        6. **주석(<!-- -->), 설명, 빈 줄 등 어떤 추가 텍스트도 삽입하지 마세요**. 출력은 오직 수정된 HTML 조각만 있어야 합니다.
        7. 최종 출력은 오직 HTML 조각이어야 하며, 다른 형식의 텍스트가 포함되면 안 됨.
        8. 최종 출력은 오직 HTML만, 다른 형식이나 설명 문구 없이 내놓으세요.
        9. 출력 언어는 한글로 해줘.
        10. **<html> <head> <body> <div> <meta> <title>** 태그는 절대 사용하지마.
        {style}
        

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        # 전체 섹션 리스트:
        {block_list.keys()}

        # 현재 섹션:
        {section_name}

        # HTML 구조 (이 구조는 변경 금지, 안에 입력 데이터 기반으로 내용을 채워넣으세요):
        {html}

        # 입력 데이터 (summary):
        {ctx_value}

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        **HTML 구조 기반으로 입력데이터 내용을 태그 규칙에 맞게 적절히 생성하여 채워 넣어서 반환할 것.**
        {style}
        **<html> <head> <body> <div> <meta> <title>** 태그는 절대 사용하지마.
        """
        gen_content = await self.send_request(content_prompt)

        gen_content = re.sub("\n", "", gen_content)
        gen_content = parser.tag_sort(gen_content)

        if not parser.validate_html_structure(gen_content, html):
            print(f"생성된 HTML 구조가 예상과 다릅니다: {section_name}")
            return {}

        return {
            'HTML_Tag': b_value,
            'Block_id': b_id,
            'gen_content': gen_content
        }

    # 기존의 extract_emmet_tag, find_key_by_value, create_tag_selection_prompt, create_content_prompt 메서드들은 그대로 유지
    def extract_emmet_tag(self, text):
        """
        주어진 문자열에서 "HTML_Tag": "..." 형태를 찾고,
        그 안의 값 중 Emmet 표기 부분(h1, h2, h3, p, li(...), etc)만 추출해서 반환합니다.
        - "신뢰도: 100%" 등 불필요한 문구가 있으면 무시하고 제거합니다.
        - 사용자가 원하는 문자만 남기고 나머지는 모두 제거합니다.
        - 유효한 Emmet 문자열이 하나도 없으면 None을 반환합니다.
        """
        # 2) 앞뒤에 있을 수 있는 별표(**) 제거
        #    예: '**h3_li(h3_p)*3**' → 'h3_li(h3_p)*3'
        raw_value = text.strip('**').strip()
        raw_value = raw_value.replace("**", "")

        # 3) 여러 줄이 있을 수 있으므로 일단 첫 줄만 취득
        #    (줄바꿈으로 split하여 첫 요소만)
        raw_value = raw_value.split('\n', 1)[0]

        # 4) '신뢰도:', '기타 불필요 텍스트' 같은 것들이 섞여 있을 수 있으니
        #    여기서는 간단히 공백 기준으로 잘라서,
        #    허용된 Emmet 문자만 남기고 나머지는 전부 제거.

        # a) 허용할 Emmet 문자를 정의 (알파벳, 숫자, 괄호, 밑줄, *, +, >, . 등)
        allowed_chars = set("hlip123456789_()*+")

        # b) raw_value 내에서 허용된 문자만 살려서 재조합
        filtered = "".join(ch for ch in raw_value if ch in allowed_chars)

        # 5) 최종 결과가 비어 있으면 None 반환
        return filtered if filtered else text
    
    def find_key_by_value(self, mapping: dict, target_value: str):
        """
        주어진 딕셔너리(mapping)에서 target_value를 값(value)으로 가지는
        첫 번째 키(key)를 찾아 반환한다. 매칭되는 키가 없으면,
        가장 유사한 값을 찾고 그에 해당하는 키를 반환한다.
        """
        # 정확히 일치하는 값 찾기
        for key, value in mapping.items():
            if value == target_value:
                return key

        # 가장 유사한 값을 찾기
        closest_match = None
        closest_score = -float('inf')  # Levenshtein 거리 점수, 큰 값일수록 유사도가 높음
        for key, value in mapping.items():
            # Levenshtein 거리 계산 (문자열 간 유사도를 0~1로 반환)
            score = difflib.SequenceMatcher(None, value, target_value).ratio()
            if score > closest_score:
                closest_match = key
                closest_score = score

        print(f"closest_match: {closest_match}, closest_score: {closest_score}")
        return closest_match if closest_match else None

    
class EmmetParser:
    def parse_emmet(self, emmet_str):
        """
        Emmet-like 문자열을 HTML로 변환하는 함수
        :param emmet_str: "h1_h2_li(h3+p)*3" 같은 문자열
        :return: 변환된 HTML 문자열
        """
        # 문자열을 '_'로 분리하여 순차적으로 처리
        parts = self.split_children(emmet_str, separator='_')
        print(f"parts : {parts}")
        html_output = ''

        for part in parts:
            html_output += self.parse_part(part)

        return html_output.strip()

    def parse_part(self, part):
        """
        단일 태그 구조를 파싱하여 HTML로 변환
        :param part: "li(h3+p)*3" 같은 문자열
        :return: 변환된 HTML 구조
        """
        # 태그, 자식, 반복 정보를 추출하는 정규식
        pattern = r'^(?P<tag>[a-z0-9]+)(?:\((?P<children>[^\)]*)\))?(?:\*(?P<count>\d+))?$'
        match = re.match(pattern, part, re.IGNORECASE)

        if not match:
            print(f"Warning: '{part}' is not a valid Emmet-like syntax.")
            return ''

        tag = match.group('tag')  # 태그명
        children = match.group('children')  # 자식 태그
        count = int(match.group('count')) if match.group('count') else 1  # 반복 횟수

        children_html = ''
        if children:
            # 자식 태그를 '+' 기준으로 분리하여 처리
            child_parts = self.split_children(children, separator='+')
            for child in child_parts:
                children_html += self.parse_part(child)

        if tag == 'li' and count > 1:
            # li는 반복 시 ul로 감싸기
            ul_content = ''
            for _ in range(count):
                ul_content += self.wrap_with_tag('li', children_html)
            return self.wrap_with_tag('ul', ul_content)
        else:
            # 일반 태그 처리
            result = ''
            for _ in range(count):
                result += self.wrap_with_tag(tag, children_html)
            return result

    def wrap_with_tag(self, tag, content):
        """
        태그로 감싸는 함수
        """
        if content:
            indented_content = self.indent_html(content)
            return f'<{tag}>\n{indented_content}</{tag}>\n'
        else:
            return f'<{tag}></{tag}>\n'

    def split_children(self, children_str, separator='_'):
        """
        문자열을 구분자로 분리하되 괄호 안의 구분자는 무시
        """
        parts = []
        current = ''
        depth = 0
        for char in children_str:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == separator and depth == 0:
                if current:
                    parts.append(current)
                    current = ''
            else:
                current += char
        if current:
            parts.append(current)
        return parts

    def indent_html(self, html_str, level=1):
        """
        HTML 들여쓰기 처리
        """
        indent = '  ' * level
        return '\n'.join([indent + line if line.strip() else line for line in html_str.split('\n')])

    def font_size(self, section_name):
        h1, h2, h3, h5, p = 10, 20, 20, 20, 20
        if section_name == 'Hero_Header':
            h1, h2, h3, h5, p = 30, 15, 30, 10, 30
        elif section_name == 'Feature' or section_name == 'Content':
            h1, h2, h3, h5, p = 10, 20, 10, 10, 30
        elif section_name == 'Testimonial' or section_name == 'Gallery':
            h1, h2, h3, h5, p = 10, 15, 10, 10, 30
        elif section_name == 'CTA':
            h1, h2, h3, h5, p = 10, 15, 10, 10, 30
        elif section_name == 'Pricing' or section_name == 'Contact' or section_name == 'Stat':
            h1, h2, h3, h5, p = 10, 15, 10, 10, 20
        elif section_name == 'Team':
            h1, h2, h3, h5, p = 10, 20, 10, 10, 30
        else:
            h1, h2, h3, h5, p = 10, 20, 10, 10, 20
        style = f'''
        다음 규칙을 엄격히 지켜 HTML 내용을 작성해.
        - <h1></h1> 태그 안의 내용은 반드시 {h1} 글자(띄어쓰기 포함) 이하로 작성할 것.  
        - <h2></h2> 태그 안의 내용은 반드시 {h2} 글자 이하로 작성할 것.  
        - <h3></h3> 태그 안의 내용은 반드시 {h3} 글자 이하로 작성할 것.
        - <h5></h5> 태그 안의 내용은 반드시 {h5} 글자 이하로 작성할 것.
        - <p></p> 태그 안의 내용은 반드시 {p} 글자 이하로 작성할 것.
        '''
        return style
    
    def tag_sort(self, gen_data):
        # 1. 마크다운 코드 블록 표시 제거 (예: ```html, ``` 등)
        cleaned = re.sub(r'```(?:\w+)?', '', gen_data)
        
        # 2. 마크다운 강조 문구 제거 (예: ***feature*** 등)
        cleaned = re.sub(r'\*{3}[^*]+\*{3}', '', cleaned)
        
        # 3. DOCTYPE 선언 제거
        cleaned = re.sub(r'<!DOCTYPE.*?>', '', cleaned, flags=re.IGNORECASE)
        
        # 4. <html>, <head>, <body>, <div> 태그와 닫는 태그 제거
        cleaned = re.sub(r'</?(html|head|body|div)(\s[^>]+)?>', '', cleaned, flags=re.IGNORECASE)
        
        # 5. <title> 태그와 그 내부 내용 및 닫는 태그 제거
        cleaned = re.sub(r'<title.*?>.*?</title>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # 6. 앞뒤 공백 제거 후 반환
        return cleaned.strip()
    def validate_html_structure(self, html_output: str, expected_structure: str) -> bool:
        """
        생성된 HTML 출력(html_output)이 예상 구조(expected_structure)의 태그들을 모두 포함하는지 검증합니다.
        내용(text)은 무시하고, 오직 태그의 존재 여부만 확인합니다.

        Args:
            html_output (str): 실제 생성된 HTML 콘텐츠
            expected_structure (str): 예상되는 HTML 구조(예: 미리 정의된 템플릿)

        Returns:
            bool: 생성된 HTML이 예상 구조대로면 True, 그렇지 않으면 False
        """
        # 예상 HTML 구조에서 모든 태그명을 추출합니다.
        expected_tags = re.findall(r'</?([a-zA-Z][a-zA-Z0-9]*)\b', expected_structure)
        # 중복 제거
        expected_tags = list(set(expected_tags))
        
        # 각 예상 태그가 생성된 HTML 내에 존재하는지 확인합니다.
        for tag in expected_tags:
            # 예: <ul>, <li>, <h1> 등 (속성은 무시)
            pattern = re.compile(rf'<{tag}(\s[^>]*?)?>', re.IGNORECASE)
            if not pattern.search(html_output):
                print(f"검증 실패: {tag} 태그가 생성된 HTML에 없습니다.")
                return False
        return True