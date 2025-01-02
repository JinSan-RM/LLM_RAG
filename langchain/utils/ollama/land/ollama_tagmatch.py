import re

def extract_body_content_with_regex(html_str: str) -> str:
    """
    정규식으로 <body>...</body> 구간만 찾아 추출.
    주의: HTML이 복잡하거나 비정상적이면 오동작할 수 있음.
    """
    # DOTALL 모드(?s)로 개행 포함해 매칭
    pattern = re.compile(r'(?s)<body[^>]*>(.*?)</body>')
    match = pattern.search(html_str)
    if not match:
        return ""  # <body>...</body> 없을 경우 빈 문자열
    
    return match.group(1)  # 그룹 1: <body>와 </body> 사이의 내용

def remove_child_ul_in_li(html_str: str) -> str:
    """
    <li> ... <ul> ... </ul> ... </li> 구조에서
    <ul>...</ul>만 정규식으로 찾아 제거.
    
    - 여러 개 <ul>이 있을 수 있으므로,
      매칭이 없을 때까지 반복 처리(while True).
    - 주의: 정규식 파싱이므로 HTML 구조가 복잡할 경우
      예상치 못한 동작이 생길 수 있음.
    """
    # <li> 태그 내부에 있는 <ul>...</ul>을 찾는 정규식
    #   (?s)  : DOTALL 모드(개행 포함 '.' 매칭)
    #   (<li>.*?)<ul>.*?</ul>(.*?</li>)
    #    -> 1) <li>로 시작, </li>로 끝나는 구간을 2덩어리로 나누어
    #       중간에 <ul>...</ul>을 제거
    pattern = re.compile(r'(?s)(<li>.*?)(<ul>.*?</ul>)(.*?</li>)')
    
    new_html = html_str
    while True:
        match = pattern.search(new_html)
        if not match:
            break
        # 그룹 1 + 그룹 3만 남기고(그룹 2: <ul>...</ul> 제거)
        new_html = pattern.sub(r'\1\3', new_html, count=1)

    return new_html


def parse_li(li_str: str, depth: int) -> str:
    """
    <li> 구간에 대해:
      - 현재 depth에서 h1/h2/h3/p를 찾아 '(...)'로 묶어서 표현
      - 만약 depth < 2라면, 자식 <ul>까지 파싱해서 이어 붙임 (3단계 무시)
      
      ※ 여기서는 이미 remove_child_ul_in_li()로 <li> 내부 <ul>이 지워져 있을 수 있으니,
        추가로 <ul> 파싱은 의미 없을 수도 있음(필요하다면 그대로 둠).
    """
    current_level_parts = []

    # (1) <h1> 개수
    h1_pattern = re.compile(r"<h1>(.*?)</h1>", re.DOTALL)
    h1_matches = h1_pattern.findall(li_str)
    if len(h1_matches) == 1:
        current_level_parts.append("h1")
    elif len(h1_matches) > 1:
        current_level_parts.append(f"(h1)*{len(h1_matches)}")

    # (2) <h2> 개수
    h2_pattern = re.compile(r"<h2>(.*?)</h2>", re.DOTALL)
    h2_matches = h2_pattern.findall(li_str)
    if len(h2_matches) == 1:
        current_level_parts.append("h2")
    elif len(h2_matches) > 1:
        current_level_parts.append(f"(h2)*{len(h2_matches)}")

    # (3) <h3> 개수
    h3_pattern = re.compile(r"<h3>(.*?)</h3>", re.DOTALL)
    h3_matches = h3_pattern.findall(li_str)
    if len(h3_matches) == 1:
        current_level_parts.append("h3")
    elif len(h3_matches) > 1:
        current_level_parts.append(f"(h3)*{len(h3_matches)}")

    # (4) <p> 개수
    p_pattern = re.compile(r"<p>(.*?)</p>", re.DOTALL)
    p_matches = p_pattern.findall(li_str)
    if len(p_matches) == 1:
        current_level_parts.append("p")
    elif len(p_matches) > 1:
        current_level_parts.append(f"(p)*{len(p_matches)}")

    # (A) 현재 <li> 태그에 모인 것들을 ( ... )로 묶기
    if current_level_parts:
        current_level_group = "(" + "_".join(current_level_parts) + ")"
    else:
        current_level_group = ""

    # (B) 자식 <ul> 처리( depth<2 ) → 실제로는 이미 제거되어 있을 가능성 높음
    children_parts = []
    if depth < 2:
        ul_pattern = re.compile(r"<ul>(.*?)</ul>", re.DOTALL)
        ul_contents = ul_pattern.findall(li_str)
        for ul_content in ul_contents:
            sub_li_list = parse_ul(ul_content, depth + 1)
            if sub_li_list:
                joined_sub = "_".join(sub_li_list)
                children_parts.append(joined_sub)

    # (C) 최종 합치기
    result_parts = []
    if current_level_group:
        result_parts.append(current_level_group)
    if children_parts:
        result_parts.append("_".join(children_parts))

    final_li_str = "_".join(result_parts)
    return final_li_str


def parse_ul(ul_str: str, depth: int) -> list:
    """
    <ul> 문자열을 받아 내부 최상위 <li>들을 찾은 뒤,
    depth 정보에 따라 parse_li(li_str, depth) 호출
    => (문자열) 리스트 반환
    """
    results = []
    li_pattern = re.compile(r"<li>(.*?)</li>", re.DOTALL)
    li_matches = li_pattern.findall(ul_str)

    for li_content in li_matches:
        li_structure = parse_li(li_content, depth)
        if li_structure:
            results.append(li_structure)

    return results


def parse_html(html_str: str) -> str:
    """
    전체 HTML 문자열을 받아서,
    1) 먼저 <li> 내부의 <ul>을 제거(정규식)
    2) 남은 <ul>/<li> 구조를 최대 depth=2까지 파싱
    3) li 안의 h1/h2/h3/p는 '( ... )' 형태로 묶어 반영
    """
    # 1) <li> 내부의 <ul>을 모두 제거
    #    (child <ul> 제거)
    cleaned_html = remove_child_ul_in_li(html_str)
    print(f"cleaned_html: {cleaned_html}")
    # 2) "최상위 <ul>" 찾고 파싱
    top_ul_pattern = re.compile(r"<ul>(.*?)</ul>", re.DOTALL)
    top_ul_contents = top_ul_pattern.findall(cleaned_html)

    all_ul_results = []
    for ul_content in top_ul_contents:
        # depth=0부터 시작
        parsed_list = parse_ul(ul_content, depth=0)
        if parsed_list:
            all_ul_results.append("_".join(parsed_list))

    # 3) 결과 연결
    final_string = "_".join(all_ul_results)
    return final_string

import re

def remove_disallowed_tags(html_str: str, allowed_tags=None) -> str:
    """
    allowed_tags: 예) ["h1","h2","h3","p","ul","li"]
    이 외의 모든 <태그>...</태그> 를 제거한다.
    
    ※ 단순화해서, <태그 ...> ~ </태그> 를 모두 제거해버리면
      태그 안의 text까지 사라지는 문제가 생길 수 있음.
    ※ '태그 자체만' 제거하고 그 내부 텍스트를 남기고 싶다면,
      다른 접근(정규식) 또는 단계별 치환이 필요.
      
    여기서는 '태그와 그 안의 내용'을 통째로 제거하는 예시로 설명.
    """
    if allowed_tags is None:
        allowed_tags = ["h1","h2","h3","p","ul","li"]
    
    # 1) 태그 이름들을 '|'로 연결
    #    예: "h1|h2|h3|p|ul|li"
    allowed_pattern = "|".join(allowed_tags)

    # 2) 허용 태그가 아닌 모든 태그를 제거하는 정규식
    #    예) <(?!/?(?:h1|h2|h3|p|ul|li)\b).*?> ... .*? </...>
    #    다만 태그 내 속성, 대소문자 등 예외처리가 많음 -> 여기선 단순화.
    
    # (A) 먼저 '대소문자 무시' 모드로 해볼 수도 있음: (?i)
    # (B) 정규식만으론 중첩 parsing 문제를 완벽 커버하기 어려움.
    
    # 여기서는 'start tag'와 'end tag'가 정확히 매칭된다고 가정한 매우 단순한 패턴 예시
    pattern_disallowed = re.compile(
        r'(?is)<(?!/?(?:' + allowed_pattern + r')\b).*?>.*?</.*?>'
    )
    
    new_html = html_str
    # 반복 치환
    while True:
        m = pattern_disallowed.search(new_html)
        if not m:
            break
        new_html = pattern_disallowed.sub("", new_html, count=1)
    
    return new_html

def remove_nested_ul(html_str: str) -> str:
    """
    <li> ... <ul> ... </ul> ... </li> 형태를 찾아
    그 <ul>...</ul> 부분을 제거해버린다.
    """
    pattern_nested_ul = re.compile(r'(?is)(<li\b[^>]*>.*?)<ul\b[^>]*>.*?</ul>(.*?</li>)')
    new_html = html_str
    while True:
        m = pattern_nested_ul.search(new_html)
        if not m:
            break
        # 그룹1 + 그룹2 => <li> 부분 + </li> 부분 사이에 <ul>... 제거
        new_html = pattern_nested_ul.sub(r'\1\2', new_html, count=1)
    return new_html

def limit_li_count_in_ul(html_str: str, max_li=5) -> str:
    """
    <ul>...</ul> 내에서 <li>가 여러 개인 경우 5개까지만 남기고 나머지는 제거.
    """
    # 1) <ul> ... </ul> 구간을 찾음
    ul_pattern = re.compile(r'(?is)<ul\b[^>]*>(.*?)</ul>')
    new_html = html_str
    
    # finditer 로 모든 <ul>...</ul> 구간
    # 각 구간 내의 <li>만 개수 제한 -> 치환해 반영
    offsets = []
    for match in ul_pattern.finditer(new_html):
        start, end = match.span()
        ul_content = match.group(1)  # <ul>과</ul> 사이 내용
        # 2) ul_content 안에 있는 <li>...</li> 를 찾는다
        li_pattern = re.compile(r'(?is)<li\b[^>]*>.*?</li>')
        li_matches = list(li_pattern.finditer(ul_content))
        
        if len(li_matches) > max_li:
            # 5개 이후는 제거
            # li들이 등장하는 순서대로, 5개 이후의 span을 없애버림
            li_spans = [ (m.start(), m.end()) for m in li_matches ]
            # 0~4번까지는 살리고, 5번 이후 삭제
            # ul_content를 list화 -> substring 조립
            keep_ranges = li_spans[:max_li]
            
            # 새 ul_content를 만들어 본다
            new_ul_content = ""
            last_index = 0
            for idx, (s, e) in enumerate(keep_ranges):
                # li 이전 substring
                new_ul_content += ul_content[last_index:s]
                # li 자체
                new_ul_content += ul_content[s:e]
                last_index = e
            # 마지막 남은 부분
            new_ul_content += ul_content[last_index:]
            
            # 치환
            # ul 전체 -> <ul> + new_ul_content + </ul>
            replaced = f"<ul>{new_ul_content}</ul>"
            
            offsets.append((start,end,replaced))
    
    # 역순 치환(오프셋 안 깨지게)
    offsets.reverse()
    for (s,e,r) in offsets:
        new_html = new_html[:s] + r + new_html[e:]
    
    return new_html

def fix_html_without_parser(html_str: str) -> str:
    # 1) 허용 태그 외 제거
    allowed_tags = ["h1","h2","h3","p","ul","li"]
    step1 = remove_disallowed_tags(html_str, allowed_tags=allowed_tags)

    # 2) 중첩 <ul> 제거
    step2 = remove_nested_ul(step1)

    # 3) <ul> 내 li가 5개 초과면 제한
    step3 = limit_li_count_in_ul(step2, max_li=5)

    # (추가) li 내부에서 h1/h2/h3/p가 아닌 것도 제거하고 싶으면
    #   별도 정규식으로 <li>...</li> 내의 허용되지 않은 태그를 찾아 제거
    #   or 다른 후속작업...
    
    return step3

def parse_ul_structure(ul_html: str) -> str:
    """
    <ul>...</ul>에서 <li>들을 찾아,
    각 <li> 내부 태그 구조가 동일하다고 가정:
      -> 첫 <li>를 parse_li_structure로 파악 => e.g. "h2_p"
      -> li 개수 N => "li(h2_p)*N"
    """
    # <li>...</li> 찾기
    li_pattern = re.compile(r'(?is)<li\b[^>]*>(.*?)</li>')
    li_contents = li_pattern.findall(ul_html)

    if not li_contents:
        return ""  # <li>가 없다면 빈

    # 1) 첫 번째 <li> 분석
    sample_li = li_contents[0]
    li_structure = parse_li_structure(sample_li)  # 예) "h2_p"

    # 2) li 개수
    count_li = len(li_contents)  # 예) 5

    if li_structure:
        # "li(h2_p)*5"
        return f"li({li_structure})*{count_li}"
    else:
        # "li()*5" 정도라도
        return f"li()*{count_li}"


def parse_li_structure(li_inner_html: str) -> str:
    """
    <li> 내부에 있는 h1/h2/h3/p 태그를 순서대로 살펴서,
    예) <h2>Title</h2><p>Paragraph</p> => "h2_p"
    여러 개 태그가 있을 수 있으므로 '_'로 연결
    """
    # 정규식으로 <h1>, <h2>, <h3>, <p> 태그를 순서대로 찾음
    pattern_tags = re.compile(r'(?is)<(h[1-3]|p)\b[^>]*>.*?</\1>')
    matches = pattern_tags.findall(li_inner_html)

    # matches 는 [("h2"), ("p"), ...] 형태
    # 하지만 group()을 직접 써야 태그 이름을 알 수 있으므로, finditer 써볼 수도 있음
    # 여기서는 간단히...
    # matches[i] -> e.g. "h2", "p"
    
    # 근데 matches를 이렇게 하면 group(1)만 나오므로, 실제 순회하면서 태그 이름 매칭
    result = []
    for m in pattern_tags.finditer(li_inner_html):
        tagname = m.group(1).lower()  # "h2", "p", ...
        result.append(tagname)

    # 예) ["h2","p"] -> "h2_p"
    return "_".join(result)

def convert_html_to_structure(html_str: str) -> str:
    """
    예) 
      <h2>Company Overview</h2>
      <ul> ... </ul>
    => "h2_li(h2_p)*5"
    """
    # (1) 최상위 태그만 순차적으로 찾기
    #     예) <h2>...</h2>, <h3>...</h3>, <p>...</p>, <ul>...</ul> 등
    #     단순히 정규식으로 "같은 레벨" 매칭 (중첩 안 되는 상황 가정)
    pattern_top = re.compile(r'(?is)(<(?:h[1-3]|p|ul)\b[^>]*>.*?</(?:h[1-3]|p|ul)>)')
    top_blocks = pattern_top.findall(html_str)

    structure_parts = []

    for block in top_blocks:
        block = block.strip()
        # h1/h2/h3/p 태그면 => "h2", "p", ...
        # ul 태그면 => parse_ul_structure -> "li(...)*N"
        tag_name = get_tag_name(block)  # 예) "h2", "ul", ...

        if tag_name in ["h1","h2","h3","p"]:
            # 그냥 "h2", "h3", "p" 등
            structure_parts.append(tag_name)
        elif tag_name == "ul":
            # ul 분석
            ul_structure = parse_ul_structure(block)
            if ul_structure:
                structure_parts.append(ul_structure)
        else:
            # 기타는 무시(사실상 없을거라 가정)
            pass

    # 최종적으로 구조 파츠를 "_"로 연결
    # 예) ["h2", "li(h2_p)*5"] -> "h2_li(h2_p)*5"
    return "_".join(structure_parts)


def get_tag_name(block: str) -> str:
    """
    예) block="<h2>Title</h2>" -> "h2"
        block="<ul>...</ul>"   -> "ul"
    단순 정규식으로 태그 이름을 뽑는다.
    """
    m = re.match(r'(?is)<(h[1-6]|p|ul)\b', block)
    if m:
        return m.group(1).lower()
    return ""  # 못 찾으면 빈 문자열