import re

def parse_li_content(li_html: str) -> dict:
    data = {}
    
    # <h2> => sub_title
    h2_pattern = re.compile(r"<h2>(.*?)</h2>", re.DOTALL)
    h2_match = h2_pattern.search(li_html)
    if h2_match:
        data["sub_title"] = h2_match.group(1).strip()

    # <h3> => strength_title
    h3_pattern = re.compile(r"<h3>(.*?)</h3>", re.DOTALL)
    h3_match = h3_pattern.search(li_html)
    if h3_match:
        data["strength_title"] = h3_match.group(1).strip()

    # <p> => description
    p_pattern = re.compile(r"<p>(.*?)</p>", re.DOTALL)
    p_match = p_pattern.search(li_html)
    if p_match:
        data["description"] = p_match.group(1).strip()

    return data


def parse_allowed_tags(html: str) -> dict:
    """
    Parse an HTML string that only contains h1, h2, h3, p, ul, li tags
    and convert it into a custom JSON/dict structure based on:
      - h1 => main_title
      - h2 => sub_titles (복수 가능)
      - h3 => strength_titles (복수 가능)
      - p  => descriptions (복수 가능)
      - ul => features (array of <li>)
    """
    result = {}

    # ──────────────────────────
    # 1) h1(main_title) 여러 개일 수도 있으니 findall 사용
    # ──────────────────────────
    h1_matches = re.findall(r"<h1>(.*?)</h1>", html, re.DOTALL)
    if h1_matches:
        # 여러 개가 있을 경우 리스트로. 하나만 있으면 [0] 사용 가능
        result["main_title"] = [m.strip() for m in h1_matches]
    else:
        print("[DEBUG] No <h1> found in HTML.")

    # ──────────────────────────
    # 2) h2 (모두)
    # ──────────────────────────
    h2_matches = re.findall(r"<h2>(.*?)</h2>", html, re.DOTALL)
    if h2_matches:
        result["sub_titles"] = [m.strip() for m in h2_matches]

    # ──────────────────────────
    # 3) h3 (모두)
    # ──────────────────────────
    h3_matches = re.findall(r"<h3>(.*?)</h3>", html, re.DOTALL)
    if h3_matches:
        result["strength_titles"] = [m.strip() for m in h3_matches]

    # ──────────────────────────
    # 4) p (모두)
    # ──────────────────────────
    p_matches = re.findall(r"<p>(.*?)</p>", html, re.DOTALL)
    if p_matches:
        result["descriptions"] = [m.strip() for m in p_matches]

    # ──────────────────────────
    # 5) ul -> li -> parse_li_content
    # ──────────────────────────
    ul_pattern = re.compile(r"<ul>(.*?)</ul>", re.DOTALL)
    ul_match = ul_pattern.search(html)

    if ul_match:
        print("[DEBUG] <ul> tag found. Parsing its contents...")  # 디버깅 출력
        result["features"] = []

        ul_content = ul_match.group(1)
        li_pattern = re.compile(r"<li>(.*?)</li>", re.DOTALL)
        li_matches = li_pattern.findall(ul_content)

        print(f"[DEBUG] Found {len(li_matches)} <li> tags inside <ul>.")  # 디버깅 출력

        for idx, li_content in enumerate(li_matches, start=1):
            print(f"[DEBUG] Parsing <li> #{idx}: {li_content.strip()}")  # 디버깅 출력
            li_data = parse_li_content(li_content)

            if li_data:
                print(f"[DEBUG] <li> #{idx} parsed result: {li_data}")  # 디버깅 출력
                result["features"].append(li_data)
            else:
                print(f"[DEBUG] <li> #{idx} had no valid sub_title/h3/p content.")
    else:
        print("[DEBUG] No <ul> found in HTML.")

    return result
