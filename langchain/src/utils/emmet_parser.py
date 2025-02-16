import re

class EmmetParser:
    def parse_emmet(self, emmet_str: str) -> str:
        parts = self.split_children(emmet_str, separator="_")
        html_output = ""
        for part in parts:
            html_output += self.parse_part(part)
        return html_output.strip()

    def parse_part(self, part: str) -> str:
        pattern = r'^(?P<tag>[a-z0-9]+)(?:\((?P<children>[^\)]*)\))?(?:\*(?P<count>\d+))?$'
        match = re.match(pattern, part, re.IGNORECASE)
        if not match:
            print(f"Warning: '{part}' is not a valid Emmet-like syntax.")
            return ''
        tag = match.group('tag')
        children = match.group('children')
        count = int(match.group('count')) if match.group('count') else 1
        children_html = ''
        if children:
            child_parts = self.split_children(children, separator='+')
            for child in child_parts:
                children_html += self.parse_part(child)
        if tag == 'li' and count > 1:
            ul_content = ''
            for _ in range(count):
                ul_content += self.wrap_with_tag('li', children_html)
            return self.wrap_with_tag('ul', ul_content)
        else:
            result = ''
            for _ in range(count):
                result += self.wrap_with_tag(tag, children_html)
            return result

    def wrap_with_tag(self, tag: str, content: str) -> str:
        if content:
            indented_content = self.indent_html(content)
            return f'<{tag}>\n{indented_content}</{tag}>\n'
        else:
            return f'<{tag}></{tag}>\n'

    def split_children(self, children_str: str, separator: str = '_') -> list:
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

    def indent_html(self, html_str: str, level: int = 1) -> str:
        indent = '  ' * level
        return '\n'.join([indent + line if line.strip() else line for line in html_str.split('\n')])

    def font_size(self, section_name: str) -> str:
        if section_name == 'Hero_Header':
            h1, h2, h3, h5, p = 30, 15, 30, 10, 30
        elif section_name in ['Feature', 'Content']:
            h1, h2, h3, h5, p = 10, 20, 10, 10, 30
        elif section_name in ['Testimonial', 'Gallery']:
            h1, h2, h3, h5, p = 10, 15, 10, 10, 30
        elif section_name == 'CTA':
            h1, h2, h3, h5, p = 10, 15, 10, 10, 30
        elif section_name in ['Pricing', 'Contact', 'Stat']:
            h1, h2, h3, h5, p = 10, 15, 10, 10, 20
        elif section_name == 'Team':
            h1, h2, h3, h5, p = 10, 20, 10, 10, 30
        else:
            h1, h2, h3, h5, p = 10, 20, 10, 10, 20
        style = f'''
        - <h1></h1>는 최대 {h1} 글자,
        - <h2></h2>는 최대 {h2} 글자,
        - <h3></h3>는 최대 {h3} 글자,
        - <h5></h5>는 최대 {h5} 글자,
        - <p></p>는 최대 {p} 글자로 작성할 것.
        '''
        return style

    def tag_sort(self, gen_data: str) -> str:
        cleaned = re.sub(r'``````', '', gen_data, flags=re.DOTALL)
        cleaned = re.sub(r'\*{3}[^*]+\*{3}', '', cleaned)
        cleaned = re.sub(r'<!DOCTYPE.*?>', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'</?(html|head|body|div)(\s[^>]+)?>', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'<title.*?>.*?</title>', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    def validate_html_structure(self, html_output: str, expected_structure: str) -> bool:
        expected_tags = re.findall(r'</?([a-zA-Z][a-zA-Z0-9]*)\b', expected_structure)
        expected_tags = list(set(expected_tags))
        for tag in expected_tags:
            pattern = re.compile(rf'<{tag}(\s[^>]*?)?>', re.IGNORECASE)
            if not pattern.search(html_output):
                print(f"검증 실패: {tag} 태그가 생성된 HTML에 없습니다.")
                return False
        return True
