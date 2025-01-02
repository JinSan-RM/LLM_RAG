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