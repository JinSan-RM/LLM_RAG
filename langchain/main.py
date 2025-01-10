'''
메인 실행 파일
'''
# main.py
from pipelines.content_chain import ContentChain
from utils.helpers import languagechecker
from utils.PDF2TXT import PDFHandle
from utils.ollama.ollama_chat import OllamaChatClient
from utils.ollama.land.ollama_menu import OllamaMenuClient
from utils.ollama.land.ollama_summary import OllamaSummaryClient
from utils.ollama.land.ollama_block_recommand import OllamaBlockRecommend
from models.models_conf import ModelParam
# local lib
# ------------------------------------------------------------------------ #
# outdoor lib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time, random
import torch, gc

app = FastAPI()


#---------------------------------
# Menu Generate Test
#---------------------------------

@app.post("/generate_menu")
async def generate_menu(path: str, path2: str='', path3: str=''):
    """
    텍스트 생성 API 엔드포인트 (스트리밍 형태로 반환)
    """
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        pdf_handle = PDFHandle()
        pdf_data = pdf_handle.PDF_request(path, path2, path3)
  
        
        start = time.time() 
        # 입력 텍스트가 한국어인지 판별 뺴야함 이부분들 한국어 체크
        discriminant = languagechecker(pdf_data)
        if discriminant:
            if len(pdf_data) > 2500:
                pdf_data = pdf_data[:2500]
        else:
            if len(pdf_data) > 8192:
                pdf_data = pdf_data[:8192]
        

        # ContentChain에서 결과 생성
        content_chain = ContentChain()
        result = content_chain.run(pdf_data, discriminant, model='bllossom', value_type='menu')
        
        print(f"Final result: {result}")  # 디버깅용 출력

        end = time.time()
        
        print("process time : ", end - start)
        return result

    except Exception as e:
        # 에러 발생 시 처리
        print(f"Error: {str(e)}")
        return {"error": str(e)}

# ======================================================================================
# LLM기반 랜딩 페이지 제작 API
# ======================================================================================

class LandPageRequest(BaseModel):
    path: str
    path2: str = ''
    path3: str = ''
    model: str = "solar"
    block: dict = {}
# Example
'''
{
    "path" : "https://cdn.zaemit.com/weven_data/app_weven/ai/PDF/회사소개서_KG이니시스.pdf",
    "path2" : "",
    "path3" : "",
    "model" : "EEVE"
    "block" : {}
}
'''
# 엔드포인트 정의
@app.post("/generate_land_section")
async def LLM_land_page_generate(request: LandPageRequest):
    """
    랜딩 페이지 섹션 생성을 처리하는 API 엔드포인트
    """
    try:
        # ========================
        #         PDF 모듈
        # ========================
        pdf_handle = PDFHandle(request.path, request.path2, request.path3)
        pdf_data = pdf_handle.PDF_request()
        
        # ========================
        #      model set 모듈
        # ========================
        model_conf = ModelParam(request.model)
        model_max_token, final_summary_length, max_tokens_per_chunk = model_conf.param_set()
        # ========================
        #      내용 요약 모듈
        # ========================
        summary_client = OllamaSummaryClient(model=request.model)
        summary = await summary_client.store_chunks(data=pdf_data, model_max_token=model_max_token, final_summary_length=final_summary_length, max_tokens_per_chunk=max_tokens_per_chunk)
        
        # ========================
        #      메뉴 생성 모듈
        # ========================
        menu_client = OllamaMenuClient(model=request.model)
        menu_structure = await menu_client.section_structure_create_logic(summary)
        # ========================
        #      블록 추천 모듈
        # ========================
        block_client = OllamaBlockRecommend(model=request.model)
        result = await block_client.generate_block_content(summary=summary, block_list=request.block)
        print(f"result : {result}")

        return result

    except Exception as e:
        print(f"Error processing landing structure: {e}")
        raise HTTPException(status_code=500, detail="Error processing landing structure.")

#===================================================================================================
#===================================================================================================

# API summary, menu, section 분리 작업. 
# pdf 읽고 summary 생성
# summary 기반으로 섹션 구조 추천
# 섹션 구조에 알맞게 context 내용 나눠서. 데이터 전송
@app.post("/land_summary_menu_generate")
async def land_summary(request: LandPageRequest):
    
    # ========================
    #         PDF 모듈
    # ========================
    pdf_handle = PDFHandle(request.path, request.path2, request.path3)
    pdf_data = pdf_handle.PDF_request()
    
    # ========================
    #      model set 모듈
    # ========================
    model_conf = ModelParam(request.model)
    model_max_token, final_summary_length, max_tokens_per_chunk = model_conf.param_set()
    
    # ========================
    #      내용 요약 모듈
    # ========================
    summary_client = OllamaSummaryClient(model=request.model)
    summary = await summary_client.store_chunks(data=pdf_data, model_max_token=model_max_token, final_summary_length=final_summary_length, max_tokens_per_chunk=max_tokens_per_chunk)

    # ========================
    #      메뉴 생성 모듈
    # ========================
    menu_client = OllamaMenuClient(model=request.model)
    section_structure, section_per_context = await menu_client.section_structure_create_logic(summary)

    # 1. 첫 번째 딕셔너리의 값들을 숫자 키 순서대로 추출
    ordered_new_keys = [section_structure[k] for k in sorted(section_structure, key=lambda x: int(x))]

    # 2. 두 번째 딕셔너리의 아이템 목록을 추출 (순서 유지)
    second_items = list(section_per_context.items())

    # 3. 순차적으로 매핑하기
    new_dict = {}
    for new_key, (_, value) in zip(ordered_new_keys, second_items):
        new_dict[new_key] = value
    print(f"result : {new_dict}")
    return summary, new_dict



# section context 데이터와 section structure 데이터를 기반으로
# 섹션 별 태그에 적합한 블럭 추천
# section 별 태그에 값을 입력

class landGen(BaseModel):
    model: str
    block: dict
    section_context: dict

# Example Data
'''
{   "model" : "EEVE",
    "block" : {
        "Navbars" : {"b101":"h1_p_p_p_p", "b111":"h2_p_p_p_p", "b121":"h3_p_p_p_p"},
        },
   "section_context" : {
        "Navbars": "KG이니시스는 1998년 설립된 전자결제 선도 기업으로, 시장 점유율 1위를 자랑하며 안전한 결제 서비스를 제공합니다. 다양한 사업 분야를 통해 온라인 및 오프라인 가맹점에 종합적인 솔루션을 제공하며 간편결제, 통합간편결제, VA(Value Added Network) 등 다양한 서비스를 제공합니다."
        }
}
'''
    
@app.post("/land_section_generate")
async def  land_section_generate(request:landGen):
    # ==============================================
    #        블록 추천 / 블록 컨텐츠 생성 모듈    
    # ==============================================
    content_client = OllamaBlockRecommend(model=request.model)
    content = await content_client.generate_block_content(block_list=request.block, context=request.section_context)
    
    return content



















#===================================================================================================

# chat 방식 테스트
@app.post("/chat_landpage_generate")
async def chat_landpage_generate(request: LandPageRequest):
    client = OllamaChatClient()
    section_options = ["Introduce", "Solution", "Features", "Social", "CTA", "Pricing", "About Us", "Team", "blog"]
    section_cnt = random.randint(6, 9)
    print(f"Selected section count: {section_cnt}")

    summary = await client.temp_store_chunks(model=request.model, data=request.input_text)
    
    # 섹션 고정 및 랜덤 채움
    section_dict = {1: "Header", 2: "Hero", section_cnt - 1: random.choice(["FAQ", "Map", "Youtube", "Contact", "Support"]), section_cnt: "Footer"}
    filled_indices = {1, 2, section_cnt - 1, section_cnt}
    for i in range(3, section_cnt):
        if i not in filled_indices:
            section_dict[i] = random.choice(section_options)
    landing_structure = dict(sorted(section_dict.items()))
    
    
    print(f"Generated landing structure: {landing_structure}")
    
    for section_num, section_name in landing_structure.items():
            print(f"Processing section {section_num}: {section_name}")

            time.sleep(0.5)
            # content = await content_client.generate_section(input_text=request.input_text, section_name=section_name)
            generated_content = await client.generate_section(model=request.model, summary=summary, section_name=section_name)
            print(f"content : {generated_content}")

    return generated_content