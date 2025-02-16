from typing import Dict, Any
from src.openai_client import send_request
from src.utils.emmet_parser import EmmetParser

async def generate_content(section_name: str, selected_block: Dict[str, Any], context: str) -> Dict[str, Any]:
    parser = EmmetParser()
    html = parser.parse_emmet(selected_block['HTML_Tag'])
    style = parser.font_size(section_name)

    content_prompt = f"""
    System: 너는 주어진 HTML 구조 내에 데이터를 채워 넣는 역할을 합니다.
    규칙:
    1. 사용자가 제공한 HTML 구조를 바꾸지 마세요.
    2. 태그 외의 추가 텍스트는 삽입하지 말 것.
    3. 출력은 오직 HTML 조각이어야 합니다.
    4. 아래 구조를 참고하세요:
    {html}
    입력 데이터: {context}

    User: HTML을 채워서 반환하세요.
    """
    
    gen_content = await send_request(content_prompt)
    gen_content = parser.tag_sort(gen_content)
    
    if not parser.validate_html_structure(gen_content, html):
        raise ValueError(f"생성된 HTML 구조가 예상과 다릅니다: {section_name}")
    
    return {
        'HTML_Tag': selected_block['HTML_Tag'],
        'Block_id': selected_block['Block_id'],
        'gen_content': gen_content
    }