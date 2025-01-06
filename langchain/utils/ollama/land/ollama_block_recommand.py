from config.config import OLLAMA_API_URL
import requests, json, re


class OllamaBlockRecommend:
    
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.4, model:str = ''):
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
            1. 내가 태그 리스트를 보여줄 건데, {summary} 내용을 토대로 {section_name}에 어울리는 태그를 하나 선정해줘.
            2. 태그들은 HTML을 부모부터 자식 순으로 해서 Emmet 문법으로 작성한 것이야.
            3. {summary} 내용을 토대로 {section_name}에 잘 어울릴 태그 목록이야 {HTMLtag_list} 이 중에서 하나를 선정해서 해줘.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            # 입력 데이터:
            {summary}
            태그 :
            {HTMLtag_list}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            반드시 단 하나의 **tag** 만 선정해서 결과를 반환하세요.
            
            """
            raw_json = await self.send_request(prompt=prompt)
            
            repeat_count = 0
            while repeat_count < 3:
                try:
                    # 1) LLM에 요청
                    # raw_json을  emmet 문법으로 뽑았는데, 이걸 다시 풀어서 쓸 수 HTML 구조로 뽑아야함.
                    section_dict = {}
                    html = self.parse_emmet(raw_json)
                    print(f"raw_json : {raw_json} || html : {html}")
                    prompt = f"""
                    <|start_header_id|>system<|end_header_id|>
                    1. {summary} 내용을 기반으로 {html} 안에 내용을 채워줘.
                    2. {summary}을 참고해서 {html} 안의 내용들을 풍부하게 채워줘.
                    3. {html} 형식을 바꿔서는 안 돼.
                    4. {summary}를 참고해서 {section_name}에 어울리는 내용으로 {html} 내용을 작성해줘.
                    <|eot_id|><|start_header_id|>user<|end_header_id|>
                    # 입력 데이터:
                    {summary}
                    섹션:
                    {html}
                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    반드시 단 하나의 **HTML** 형태로 결과를 반환하세요.
                    """
                    print(f"len prompt : {len(prompt)}")
                    gen_content = await self.send_request(prompt=prompt)

                    gen_content = re.sub("\n", "", gen_content)
                    print(f"raw_json : {type(gen_content)} {len(gen_content)} / {gen_content}")
                    section_dict['HTML_Tag'] = raw_json
                    section_dict['gen_content'] = gen_content
                    result_dict['section_name'] = section_dict
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
