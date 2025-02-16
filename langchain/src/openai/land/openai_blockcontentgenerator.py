from typing import Dict, Any
from src.openai_client import send_request
from src.utils.emmet_parser import EmmetParser




async def generate_content(section_name: str, selected_block: Dict[str, Any], section_context: str) -> Dict[str, Any]:
    parser = EmmetParser()
    html = parser.parse_emmet(selected_block['HTML_Tag'])
    style = parser.font_size(section_name)

    """content_prompt = f
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
    # 이걸 사용함에 있어서 h5는 우리만 아는 규칙이므로 h5는 치환해서 사용할 필요가 있음
    # LLM에서 얘기하는건 <span class = "subtitle">이므로, 이걸 좀 쉽게 만들어서 subtitle로 치환.
    # h5 -> subtitle
    
    # NOTE : Q. li는 넣을 필요 없죠?
    # NOTE : Q. 아래에 Example은 제가 잘못 적었을 수도 있어요! 얘기하면서 저희 형식에 맞춰봅시다 :)
    content_prompt = f"""
    System: 
    You are an AI assistant that uses Section_context to create content for each semantic tag.
    If the user provides Section_context and json_type_input, read the content first and then create content to enter the semantic tag in each key.
    
    #### Explain Semantic Tag ####
    h1: Represents the most important title of the web page. Typically only one is used per page, and it represents the topic or purpose of the page.
    h2: Indicates the next most important heading after h1. It is mainly used to separate major sections of a page.
    h3: A subheading of h2, indicating detailed topics within the section.
    subtitle : It is located around h tags and briefly assists them.
    p: A tag that defines a paragraph. Used to group plain text content.
    li: Represents a list item. Mainly used within ul (unordered list) or ol (ordered list) tags.
    
    #### Instructions ####
    
    1. Do not change the json structure provided by the user.
    2. Do not insert additional text other than the value of each key.
    3. ensure that the output mathces the JSON output example below.
    
    #### Example JSON Output ####
\\\\\\`json {{ "h1": "description", 
                "h2" : "description",
                "p" : "Description", 
                "li" : [
                    "h2" : "description",
                    "p" : "Description",
                    "h2" : "description",
                    "p" : "Description",
                    ]
                    }}

    User: 
    json_type_input : {}
    Section_context : {section_context}
    
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