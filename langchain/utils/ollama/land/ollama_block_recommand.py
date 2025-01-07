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
            
            prompt = f"""
            <|start_header_id|>system<|end_header_id|>
            1. {section_name} 섹션에 가장 잘 어울리는 태그 하나를 태그리스트트 중에서 선정하세요.
            2. 단 하나의 태그만 반환하시고, 그 외의 어떤 텍스트도 출력하지 마세요.
            3. 태그의 HTML 구조는 변경하지 말고 그대로 출력하세요.
            4. **주석(<!-- -->), 설명, 빈 줄 등 추가 텍스트나 요약본, 문장(해설, 코드 등)을 삽입하지 마세요.**
            5. **태그리스트 중 하나만** 최종 출력해야 하며, 다른 모든 형식(JSON, 코드 블록 등)은 절대 포함하지 마세요.
            6. 예: {HTMLtag_list[0]}
            7. 만약 태그 리스트 중 하나가 아닌 다른 텍스트가 발견되면, 오류가 발생했다고 판단하고 **재시도**하세요.
            8. 출력예시를 따라 출력하세요.

            <|eot_id|><|start_header_id|>user<|end_header_id|>

            # 태그 리스트:
            {HTMLtag_list}

            출력예시:
            {HTMLtag_list[0]}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            **반드시 태그 리스트 안의 데이터 중 하나만 골라서 그 값만 출력하세요.**
            **최종 출력은 오직 태그만 있어야 하며, 다른 형식(JSON, 코드 블록, 설명 등)이나 텍스트를 절대 포함하면 안 됩니다.**
            """
            
            raw_json = await self.send_request(prompt=prompt)
            print(f"raw_json bf: {raw_json}")
            raw_json = self.extract_emmet_tag(raw_json)
            print(f"raw_json af: {raw_json}")
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
                    아래 지침을 철저히 지키세요:

                    1. 사용자에게서 입력받은 HTML 구조(html)는 절대 변경하지 않습니다.
                    - 태그 이름, 태그 계층, 태그의 개수, 순서, 닫힘 태그 등 **그대로 유지**해야 합니다.
                    - 예: <ul> 내부에 li가 3개면, 꼭 3개만 있어야 하고 추가/삭제 불가.
                    - 다른 태그나 속성(class, style, id 등)을 추가하거나 삭제하지 마세요.

                    2. 입력 데이터(summary)를 바탕으로, **태그 안의 내용(텍스트)만 적절히 채워넣어** 주세요.
                    - 예: <h1> </h1> 사이에 summary에서 추출한 내용 등을 넣어, <h1>최종 문구</h1> 형태로 완성합니다.
                    - 필요하다면 문장 요약/재구성해서 각 태그 안에 할당하지만, **구조(태그)는 건드리지 않습니다**.

                    3. **주석(<!-- -->), 설명, 빈 줄 등 어떤 추가 텍스트도 삽입하지 마세요**.
                    - 출력은 오직 수정된 HTML 조각만 있어야 합니다.

                    4. **아무런 HTML 속성도 붙이지 말 것** (예: class, style, id 등 금지).
                    
                    5. <html>, <head>, <body>, <!DOCTYPE> 같은 최상위 태그는 **절대 추가하지 마세요.**

                    6. 최종 출력은 JSON, 코드블록, 설명, 따옴표 등을 쓰지 말고, **오직 HTML**로만 내놓으세요.


                    <|eot_id|><|start_header_id|>user<|end_header_id|>
                    # 입력 데이터 (summary):
                    {summary}

                    # HTML 구조 (이 구조는 변경 금지, 안에 텍스트만 채워넣으세요):
                    {html}

                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    - 위 두 정보를 사용해, **주어진 HTML 구조** 안쪽에만 summary 내용을 적절히 배치하세요.
                    - <html>, <head>, <body> 같은 태그를 추가로 만들지 마세요.
                    - 구조(태그 이름, 계층, 태그 수, 순서)는 절대 바꾸지 말고, **속성·주석·추가 태그 없이** 그대로 출력하세요.
                    - 최종 출력은 오직 HTML만, 다른 형식이나 설명 문구 없이 내놓으세요.
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
        raw_value = raw_value.replace("**","")

        # 3) 여러 줄이 있을 수 있으므로 일단 첫 줄만 취득
        #    (줄바꿈으로 split하여 첫 요소만)
        raw_value = raw_value.split('\n', 1)[0]

        # 4) '신뢰도:', '기타 불필요 텍스트' 같은 것들이 섞여 있을 수 있으니
        #    여기서는 간단히 공백 기준으로 잘라서,
        #    허용된 Emmet 문자만 남기고 나머지는 전부 제거.
        
        # a) 허용할 Emmet 문자를 정의 (알파벳, 숫자, 괄호, 밑줄, *, +, >, . 등)
        allowed_chars = set("hlip123456789_()*")
        

        # b) raw_value 내에서 허용된 문자만 살려서 재조합
        filtered = "".join(ch for ch in raw_value if ch in allowed_chars)

        # 5) 최종 결과가 비어 있으면 None 반환
        return filtered if filtered else text

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
