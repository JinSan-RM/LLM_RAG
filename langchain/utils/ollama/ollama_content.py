import requests, json, random, re
from config.config import OLLAMA_API_URL
from fastapi import HTTPException
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import tiktoken
from typing import List
from utils.ollama.ollama_embedding import get_embedding_from_ollama

class OllamaContentClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.4, structure_limit = True,  n_ctx = 4096, max_token = 4096):
        self.api_url = api_url
        self.temperature = temperature
        self.structure_limit = structure_limit
        self.n_ctx = n_ctx
        self.max_token = max_token
        
    async def send_request(self, model: str, prompt: str) -> str:
        """
        공통 요청 처리 함수: API 호출 및 응답 처리
        """
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": self.temperature,
            "n_ctx": self.n_ctx,
            "repetition penalty":1.2,
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
    #========================================================================================
    # chunk test code
    

































































































































































































    #========================================================================================  
    # async def contents_GEN(self, model : str= "bllossom", input_text = "", section_name=""):
        
    #     prompt = f"""
    #             <|start_header_id|>system<|end_header_id|>
    #             - 너는 사이트의 섹션 구조를 정해주고, 그 안에 들어갈 내용을 작성해주는 AI 도우미야.
    #             - 입력된 데이터를 기준으로 단일 페이지를 갖는 랜딩사이트 콘텐츠를 생성해야 해.
    #             - 'children'의 컨텐츠 내용의 수는 너가 생각하기에 섹션에 알맞게 개수를 수정해서 생성해줘.
    #             - 섹션 '{section_name}'에 어울리는 내용을 생성해야 하며, 반드시 다음 규칙을 따라야 한다:
    #             1. assistant처럼 생성해야하고 형식을을 **절대** 벗어나면 안 된다.
    #             2. "div, h1, h2, h3, p, ul, li" 태그만 사용해서 섹션의 콘텐츠를 구성해라.
    #             3. 섹션 안의 `children` 안의 컨텐츠 개수는 2~10개 사이에서 자유롭게 선택하되, 내용이 반복되지 않도록 다양하게 생성하라.
    #             4. 모든 텍스트 내용은 입력 데이터에 맞게 작성하고, 섹션의 목적과 흐름에 맞춰야 한다.
    #             5. 출력 결과는 코드 형태만 허용된다. 코드는 **절대 생성하지 마라.**
    #             6. 오직 한글로만 작성하라.
                

    #             <|eot_id|><|start_header_id|>user<|end_header_id|>
    #             섹션:
    #             {section_name}
                

    #             <|eot_id|><|start_header_id|>assistant<|end_header_id|>

    #             <|eot_id|><|start_header_id|>user<|end_header_id|>
                
                

    #             <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    #             - 너는 코드 구조 응답만을 반환해야 한다.
    #             """
    #     # prompt = f"""
    #     #         <|start_header_id|>system<|end_header_id|>
    #     #         - 너는 사이트의 섹션 구조를 정해주고, 그 안에 들어갈 내용을 작성해주는 AI 도우미야.
    #     #         - 입력된 데이터를 기준으로 단일 페이지를 갖는 랜딩사이트 콘텐츠를 생성해야 해.
    #     #         - 'children'의 컨텐츠 내용의 수는 너가 생각하기에 섹션에 알맞게 개수를 수정해서 생성해줘.
    #     #         - 섹션 '{section_name}'에 어울리는 내용을 생성해야 하며, 반드시 다음 규칙을 따라야 한다:
    #     #         1. assistant처럼 생성해야하고 형식을을 **절대** 벗어나면 안 된다.
    #     #         2. "h1, h2, h3, p" 태그만 사용해서 섹션의 콘텐츠를 구성해라.
    #     #         3. 섹션 안의 `children` 안의 컨텐츠 개수는 2~10개 사이에서 자유롭게 선택하되, 내용이 반복되지 않도록 다양하게 생성하라.
    #     #         4. 모든 텍스트 내용은 입력 데이터에 맞게 작성하고, 섹션의 목적과 흐름에 맞춰야 한다.
    #     #         5. 출력 결과는 JSON 형태만 허용된다. 코드는 **절대 생성하지 마라.**
    #     #         6. 오직 한글로만 작성하라.
                

    #     #         <|eot_id|><|start_header_id|>user<|end_header_id|>
    #     #         입력 데이터:
    #     #         {input_text}
    #     #         섹션:
    #     #         {section_name}
                

    #     #         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    #     #         - 너는 JSON 형태의 응답만을 반환해야 한다. 아래와 같은 형식의 순수 JSON만을 출력해야해.
    #     #         {{
    #     #             "children": [
    #     #                 {{
    #     #                     "type": "h1",
    #     #                     "text": "섹션의 주요 제목을 입력 데이터에 맞게 작성합니다."
    #     #                 }},
    #     #                 {{
    #     #                     "type": "p",
    #     #                     "text": "섹션의 내용을 소개하는 첫 번째 단락입니다. 핵심 메시지를 간결하고 명확하게 작성합니다."
    #     #                 }},

    #     #                 {{
    #     #                     "type": "h3",
    #     #                     "text": "중요 포인트를 정리하거나 강조합니다."
    #     #                 }},
    #     #                 {{
    #     #                     "type": "p",
    #     #                     "text": "설명 내용을 구체적으로 작성하되 중복되지 않도록 주의합니다."
    #     #                 }},
    #     #                 {{
    #     #                     "type": "p",
    #     #                     "text": "마무리 문장으로 섹션의 가치를 강조하고 독자의 참여를 유도합니다."
    #     #                 }}
    #     #             ]
    #     #         }}
    #     #         """
    #     return await self.send_request(model, prompt)
    
    # #============================================================================
    # # test code 칸 chunk
    # async def send_pdf_chunk(self, model: str, total_chunks, current_chunk_number, chunk_content: str) -> str:
    #     """
    #     PDF 청크를 보내어 세션에 누적
    #     """
    #     prompt = f"""
    #         <|start_header_id|>system<|end_header_id|>
    #         - 다음은 PDF 문서의 일부입니다. 이 내용을 기억하고, 이후의 질문에 참고해 주세요.
    #         - 문서는 총 {total_chunks}개의 청크로 나누어져 있습니다.
    #         - 현재 제공되는 청크는 {current_chunk_number}번째 청크입니다.
    #         - 각 청크는 고유한 식별자를 가지고 있으며, 필요 시 해당 청크를 참조할 수 있습니다.
            
    #         <|eot_id|><|start_header_id|>user<|end_header_id|>
    #         - Chunk {current_chunk_number} of {total_chunks}:
    #         - {chunk_content}
    #     """
    #     return await self.send_request(model, prompt)
    # #============================================================================
    
    # async def landing_block_STD(self, model : str= "bllossom", input_text :str = "", section_name=""):
    #     prompt = f"""
    #         <|start_header_id|>system<|end_header_id|>
    #         - 당신은 AI 랜딩페이지 콘텐츠 작성 도우미입니다.
    #         - 입력된 데이터를 기반으로 랜딩페이지의 적합한 콘텐츠를 작성하세요.
    #         - 반드시 입력 데이터를 기반으로 작성하며, 추가적인 내용은 절대 생성하지 마세요.
    #         - 섹션에 이름에 해당하는 내용 구성들로 내용 생성하세요.
    #         - 콘텐츠를 JSON 형태로 작성하세요.

    #         <|eot_id|><|start_header_id|>user<|end_header_id|>
    #         입력 데이터:
    #         {input_text}
            
    #         섹션:
    #         {section_name}

    #         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    #         - 출력형식을 제외하고 다른 정보는 출력하지마세요.
    #         - 출력은 JSON형태로만 출력하세요.
    #         **출력 예시**:
    #         {{"h1" : "타이틀 내용",
    #         "h2" : (선택사항)"서브타이틀 내용",
    #         "h3" : 
    #         "본문" : "본문내용"}}
    #         """
                
    #     print(f"prompt length : {len(prompt)}")
    #     return await self.send_request(model, prompt)
    
    # async def LLM_content_fill(self, input_text: str = "", model="bllossom", summary = ""):
        
    #     prompt = f"""
    #             <|start_header_id|>system<|end_header_id|>
    #             당신은 전문적이고 매력적인 랜딩페이지 컨텐츠를 생성하는 고급 AI 어시스턴트입니다. 다음 지침을 철저히 따르세요:

    #             **주요 목표:**
    #             - 제공된 입력 데이터와 요약 데이터를 기반으로 컨텐츠를 작성하세요.
    #             - 작성된 컨텐츠는 타겟 고객의 관심을 끌 수 있도록 매력적이어야 합니다.

    #             **작성 지침:**
    #             - 모든 응답은 반드시 한글로 작성하세요.
    #             - 각 섹션의 형식을 유지하며 내용을 작성하세요.

    #             <|eot_id|><|start_header_id|>user<|end_header_id|>
    #             입력 데이터:
    #             {input_text}

    #             <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    #             - 요약 데이터를 바탕으로 입력 데이터에서 필요한 내용을 도출하여 작성합니다.
    #             - 아래와 같은 형식으로 컨텐츠를 구성합니다:

    #             1. 입력 데이터의 모든 중요 정보 포함
    #             2. 최종 컨텐츠는 명확하고, 설득력 있으며, 전문성을 갖추도록 작성

    #             주의사항:
    #             - 문법적 오류와 부자연스러운 표현 주의
    #     """
    #     print(f"LLM_content_fill Len :  {len(prompt)}")
    #     return await self.send_request(model, prompt)
    
    
    # async def LLM_land_page_content_Gen(self):
    #     """
    #     랜딩 페이지 섹션을 생성하고 JSON 구조로 반환합니다.
    #     """
    #     # 섹션 리스트
    #     section_options = ["Introduce", "Solution", "Features", "Social", 
    #                     "CTA", "Pricing", "About Us", "Team","blog"]

    #     # 섹션 수 결정 (6 ~ 9개)
    #     section_cnt = random.randint(6, 9)
    #     print(f"Selected section count: {section_cnt}")

    #     # 1번과 2번 섹션은 고정
    #     section_dict = {
    #         1: "Header",
    #         2: "Hero"
    #     }

    #     # 마지막 섹션은 Footer로 고정
    #     section_dict[section_cnt] = "Footer"

    #     # 마지막 이전 섹션에 FAQ, Map, Youtube 중 하나 배정
    #     minus_one_sections = ["FAQ", "Map", "Youtube", "Contact", "Support"]
    #     section_dict[section_cnt - 1] = random.choice(minus_one_sections)

    #     # 나머지 섹션을 랜덤하게 채움
    #     filled_indices = {1, 2, section_cnt - 1, section_cnt}
    #     for i in range(3, section_cnt):
    #         if i not in filled_indices:
    #             section_dict[i] = random.choice(section_options)

    #     # 섹션 번호 순서대로 정렬
    #     sorted_section_dict = dict(sorted(section_dict.items()))

    #     # JSON 문자열 반환
    #     result_json = json.dumps(sorted_section_dict, indent=4)
    #     print("Generated Landing Page Structure:")
    #     print(result_json)
    #     return result_json
    
    


    # async def LLM_land_block_content_Gen(self, input_text : str = "", model = "bllossom", section_name = "", section_num = "1", summary=""):
        
    #     try:
    #         # 비동기 함수 호출 시 await 사용
    #         contents_data = await self.landing_block_STD(model=model, input_text=input_text, section_name = section_name, section_num = section_num)
    #         print(f"contents_data summary before: {contents_data}")
            
    #         # 최종 수정된 딕셔너리를 JSON 문자열로 변환하여 반환
    #         contents_data = await self.LLM_content_fill(model=model, input_text=contents_data, summary=summary)
    #         print(f"contents_data summary after: {contents_data}")
            
    #         return contents_data
    #     except Exception as e:
    #         print(f"Error generating landing page sections: {e}")
    #         raise HTTPException(status_code=500, detail="Failed to generate landing page sections.")