from config.config import OLLAMA_API_URL
import requests, json, re


class OllamaBlockRecommend:
    
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.25, model:str = ''):
        self.api_url = api_url
        self.temperature = temperature
        self.model = model
    
    async def send_request(self, prompt: str) -> str:
        """
        공통 요청 처리 함수 : API 호출 및 응답 처리
        
        Generate 버전전
        """
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리

            full_response = response.text  # 전체 응답
            lines = full_response.splitlines()
            all_text = ""
            for line in lines:
                try:
                    json_line = json.loads(line.strip())  # 각 줄을 JSON 파싱
                    all_text += json_line.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    continue  # JSON 파싱 오류 시 건너뛰기
                
            return all_text.strip() if all_text else "Empty response received"

        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 실패: {e}")
            raise RuntimeError(f"Ollama API 요청 실패: {e}")
        
    async def generate_block_content(self, summary:str, block_list: dict ):
        """
        랜딩 페이지 섹션을 생성하는 함수
        """

        # 블록 리스트들을 받아와서 summary 데이터랑 합쳐서 가장 적합할 블록 추천해달라는 근데 섹션 전체에 각각 다.
        # 프롬프트
        result_dict = {}
        for section_name, HTMLtag_list in block_list.items():
            
            prompt=f"""
            <|start_header_id|>system<|end_header_id|>
            1. 입력 데이터 내용을 바탕으로 {section_name} 섹션에 가장 어울리는 태그 하나를 {HTMLtag_list} 중에서 선정해 주세요.
            2. 태그의 HTML 구조는 변경하지 말고, 단 하나의 **HTML 태그**만 반환하세요.
            3. **반드시 단 하나의 Emmet 형식의 HTML 태그**만 출력하고, 그 외의 어떤 텍스트도 포함하지 마세요.
            4. 주석(<!-- -->), 설명, 빈 줄 등 모든 추가 텍스트를 금지합니다.
            5. 어떤 설명(문장, 코드, 해설, "")도 삽입하지 말 것.
            6. 최종 출력은 오직 Emmet 형식의 HTML 태그만 있어야 하며, 다른 형식(JSON, 코드 블록 등)은 사용하지 마세요.
            7. 출력예시 대로 출력하세요.
            
            <|end_system|>

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            # 입력 데이터:
            {summary}

            # 태그 리스트:
            {HTMLtag_list}
            
            출력예시:
            
            {HTMLtag_list[0]}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            **반드시 태그 리스트 안의 데이터중 하나를 골라서 그 값만 반환하세요.**
            최종 출력은 오직 데이터만 있어야 하며, 다른 형식(JSON, 코드 블록, 설명 등)은 사용하지 마세요.
            """

            raw_json = await self.send_request(prompt=prompt)
            
            repeat_count = 0
            while repeat_count < 3:
                try:
                    # 1) LLM에 요청
                    # raw_json을  emmet 문법으로 뽑았는데, 이걸 다시 풀어서 쓸 수 HTML 구조로 뽑아야함.
                    section_dict = {}
                    parser = EmmetParser()
                    print(raw_json)
                    html = parser.parse_emmet(raw_json)
                    print(f"raw_json : {raw_json} || html : {html}")
                    prompt = f"""
                    <|start_header_id|>system<|end_header_id|>
                    1. 제공된 입력 데이터 내용을 바탕으로 {section_name} 섹션 내에 적절한 내용을 채워주세요.
                    2. HTML 구조는 변경하지 않고, 기존의 태그와 형식을 유지하면서 내용을 풍부하게 작성해주세요.
                    3. 제공받은 html 구조의 컨텐츠를 채우는 것 외에 다른 설명, 첨부, 주석등을 하지 마세요.
                    **아무런 속성도 붙이지 말 것** (class, style, id 등 css 불가)  
                    주석(<!-- -->)이나 설명, 빈 줄 등 추가 텍스트 전부 금지
                    최종 출력은 별도의 텍스트 설명이나 JSON, 기타 형식은 넣지 않는다. 오직 HTML만 출력한다.
                    **결과는 제공받은 섹션의 구조안의 내용만 채운다.**
                    <|eot_id|><|start_header_id|>user<|end_header_id|>
                    # 입력 데이터:
                    {summary}

                    # 섹션:
                    {html}
                    
                    출력예시:
                    
                    <h1> content title </h1>
                    <h3> strength title </h3>
                    <ul>
                        <li>
                            <h2> sub title </h2>
                            <p> description </p>
                        </li>
                        <li>
                            <h2> sub title2 </h2>
                            <p> description2 </p>
                        </li>
                        <li>
                            <h2> sub title3 </h2>
                            <p> description3 </p>
                        </li>
                        <li>
                            <h2> sub title4 </h2>
                            <p> description4 </p>
                        </li>
                    </ul>
                    <|end_user|>

                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    - 입력데이터와 섹션을 기반으로 **HTML**형태 결과를 반환하세요.
                    반드시 단 하나의 완전한 **HTML** 형태로 결과를 반환하세요.
                    **아무런 속성도 붙이지 말 것** (class, style, id 등 css 불가) 
                    """
                    print(f"len prompt : {len(prompt)}")
                    gen_content = await self.send_request(prompt=prompt)

                    gen_content = re.sub("\n", "", gen_content)
                    print(f"raw_json : {type(gen_content)} {len(gen_content)} / {gen_content}")
                    section_dict['HTML_Tag'] = raw_json
                    section_dict['gen_content'] = gen_content
                    result_dict[f'{section_name}'] = section_dict
                    break
                except RuntimeError as r:
                    print(f"Runtime error: {r}")
                    repeat_count += 1
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    repeat_count += 1
                    
        return result_dict
    
    
    # ============================================================
class EmmetParser:
    def parse_emmet(self, emmet_str):
        """
        Emmet-like 문자열을 HTML로 변환하는 함수
        :param emmet_str: 예: "h1_h1_h3_li(h2_h3)*2"
        :return: 변환된 HTML 문자열
        """
        # 최상위 '_' 단위로 분리
        top_level_parts = self.split_children(emmet_str)
        html_output = ''

        for part in top_level_parts:
            html_output += self.parse_part(part)

        return html_output.strip()

    def parse_part(self, part):
        """
        단일 파트(예: "li(h2_h3)*2")를 HTML로 변환하는 재귀 함수
        :param part: "li(h2_h3)*2" 같이 괄호, 반복 등이 섞인 문자열
        :return: 변환된 HTML 문자열
        """
        # 정규식을 통해 태그명, 자식, 반복 횟수 등을 추출
        pattern = r'^(?P<tag>[a-z0-9]+)(?:\((?P<children>[^\)]*)\))?(?:\*(?P<count>\d+))?$'
        match = re.match(pattern, part, re.IGNORECASE)

        if not match:
            # 매칭되지 않으면 빈 문자열 반환 또는 예외 처리
            print(f"Warning: '{part}' is not a valid Emmet-like syntax.")
            return ''

        tag = match.group('tag')
        children = match.group('children')
        count = int(match.group('count')) if match.group('count') else 1

        # 자식 요소가 있는 경우 재귀적으로 파싱
        children_html = ''
        if children:
            child_parts = self.split_children(children)
            for child in child_parts:
                children_html += self.parse_part(child)

        # 반복 처리
        if tag == 'li' and count > 1:
            # li를 반복하면 <ul>로 감싸준다
            ul_content = ''
            for _ in range(count):
                ul_content += self.wrap_with_tag('li', children_html)
            return self.wrap_with_tag('ul', ul_content)
        else:
            # 그 외 태그들은 단순히 반복해서 붙여준다
            result = ''
            for _ in range(count):
                result += self.wrap_with_tag(tag, children_html)
            return result

    def wrap_with_tag(self, tag, content):
        """
        태그로 감싸는 함수
        :param tag: 태그명 (예: 'h1', 'p', 'ul' 등)
        :param content: 태그 내부에 들어갈 내용
        :return: 감싸진 HTML 문자열
        """
        if content:
            # 들여쓰기를 추가하여 가독성을 높임
            indented_content = self.indent_html(content)
            return f'<{tag}>\n{indented_content}</{tag}>\n'
        else:
            return f'<{tag}></{tag}>\n'

    def split_children(self, children_str):
        """
        자식 요소 문자열을 '_' 단위로 분리하되, 괄호 내의 '_'는 무시하는 함수
        :param children_str: "h2_h3" 등
        :return: 분리된 자식 요소 리스트
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
            elif char == '_' and depth == 0:
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
        HTML 문자열에 들여쓰기를 추가하는 함수
        :param html_str: 들여쓰기를 추가할 HTML 문자열
        :param level: 기본 들여쓰기 레벨
        :return: 들여쓰기가 적용된 HTML 문자열
        """
        indent = '  ' * level
        indented = ''.join([indent + line if line.strip() else line for line in html_str.split('\n')])
        return indented
