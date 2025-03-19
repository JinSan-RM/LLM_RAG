'''
메인 실행 파일
'''
# main.py
from pipelines.content_chain import ContentChain
from utils.PDF2TXT import PDFHandle
from utils.ollama.land.ollama_menu import OllamaMenuClient
from utils.ollama.land.ollama_summary import OllamaSummaryClient
from utils.ollama.land.ollama_block_recommand import OllamaBlockRecommend
from utils.milvus_collection import MilvusDataHandler
from utils.ollama.land.ollama_contents_merge import OllamaDataMergeClient
from utils.ollama.land.ollama_examine import OllamaExamineClient
from utils.ollama.land.ollama_keyword import OllamaKeywordClient
from utils.ollama.land.ollama_usr_data_argument import OllamaUsrMsgClient
from models.models_conf import ModelParam
# from utils.imagine_gen import AugmentHandle
# from utils.ollama.ollama_chat import OllamaChatClient

from src.configs.call_config import Completions
from src.configs.openai_config import OpenAIConfig
from src.openai.openai_api_call import OpenAIService
from src.utils.batch_handler import BatchRequestHandler

from src.openai.land.openai_usrmsgclient import OpenAIUsrMsgClient
# from src.openai.land.openai_pdfsummary import OpenAIPDFSummaryClient, OpenAIComprehensiveProposalClient, OpenAIProposalClient
# from src.openai.land.openai_pdfsummary import OpenAIPDFSummaryClient, OpenAIProposalClient
from src.openai.land.openai_pdfsummary import OpenAIProposalClient
# from src.openai.land.openai_usrpdfmerge import OpenAIDataMergeClient
from src.openai.land.openai_usrpdfmerge import OpenAIComprehensiveProposalClient
from src.openai.land.openai_sectiongenerator import OpenAISectionGenerator
from src.openai.land.openai_blockrecommend import OpenAIBlockSelector
from src.openai.land.openai_blockcontentgenerator import OpenAIBlockContentGenerator
from src.openai.land.openai_keywordforimage import OpenAIKeywordClient


# local lib
# ------------------------------------------------------------------------ #
# outdoor lib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import torch
import gc
from pymilvus import Collection
from typing import List
import logging
import asyncio
# import random


app = FastAPI()


# ---------------------------------
# Menu Generate Test
# ---------------------------------

@app.post("/generate_menu")
async def generate_menu(path: str = '', path2: str = '', path3: str = ''):
    """
    텍스트 생성 API 엔드포인트 (스트리밍 형태로 반환)
    """
    gc.collect()
    torch.cuda.empty_cache()
    try:
        pdf_handle = PDFHandle(path=path, path2=path2, path3=path3)
        pdf_data = pdf_handle.PDF_request()
        start = time.time()
        # ContentChain에서 결과 생성
        content_chain = ContentChain()
        result = content_chain.run(
            pdf_data,
            model='bllossom',
            value_type='menu'
            )
        print(f"Final result: {result}")  # 디버깅용 출력
        end = time.time()
        print("process time : ", end - start)
        return result
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
    except ValueError as e:
        print(f"값 오류: {e}")
    except Exception as e:
        print(f"Error: {str(e)}")  # 에러 발생 시 처리
        return {"error": str(e)}

# ======================================================================================
# LLM기반 랜딩 페이지 제작 API
# ======================================================================================


class LandPageRequest(BaseModel):
    path: str = ''
    path2: str = ''
    path3: str = ''
    model: str = ''
    block: dict = {}
    user_msg: str = ''


# Example
Example = '''
{
    "path" : "https://cdn.zaemit.com/weven_data/app_weven/ai/PDF/회사소개서_KG",
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
        #    비속어 욕 체크 모듈
        # ========================
        examine_client = OllamaExamineClient(
            model=request.model,
            data=request.user_msg
            )
        examine = await examine_client.data_examine()
        if examine in "비속어":
            return "1"
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
        print(model_max_token, final_summary_length, max_tokens_per_chunk)
        # ========================
        #      내용 요약 모듈
        # ========================
        summary_client = OllamaSummaryClient(model=request.model)
        summary = await summary_client.store_chunks(
            data=pdf_data,
            model_max_token=model_max_token,
            final_summary_length=final_summary_length,
            max_tokens_per_chunk=max_tokens_per_chunk
            )
        # ========================
        #    비속어 욕 체크 모듈
        # ========================
        examine_client = OllamaExamineClient(
            model=request.model,
            data=summary
            )
        examine = await examine_client.data_examine()
        if examine in "비속어":
            return "1"

        contents_client = OllamaDataMergeClient(
            model=request.model,
            user_msg=request.user_msg,
            data=summary
            )

        summary = await contents_client.contents_merge()
        # ========================
        #      메뉴 생성 모듈
        # ========================
        menu_client = OllamaMenuClient(model=request.model)
        section_structure, section_per_context = await menu_client.section_structure_create_logic(summary)
        print(section_structure, section_per_context)

        # 1. 첫 번째 딕셔너리의 값들을 숫자 키 순서대로 추출
        ordered_new_keys = [section_structure[k] for k in sorted(section_structure, key=int)]
        section_structure_copy = ordered_new_keys.copy()
        ordered_new_keys.insert(0, "Header")
        ordered_new_keys.append("Footer")
        print(f"ordered_new_keys : {ordered_new_keys}")

        # 2. 두 번째 딕셔너리(객체)의 아이템 목록을 추출 (순서 유지)
        second_items = list(section_per_context.items())
        second_items.insert(0, ('Header', ', '.join(section_structure_copy)))
        second_items.append(('Footer', ', '.join(section_structure_copy)))
        print(f"second_items : {second_items}")

        # 3. 순차적으로 매핑하기
        new_dict = {}
        for new_key, (_, value) in zip(ordered_new_keys, second_items):
            new_dict[new_key] = value

        return summary, new_dict

    except Exception as e:
        print(f"Error processing landing structure: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error processing landing structure."
        ) from e

# ===================================================================================================
# API summary, menu, section 분리 작업.
# pdf 읽고 summary 생성
# summary 기반으로 섹션 구조 추천
# 섹션 구조에 알맞게 context 내용 나눠서. 데이터 전송
valid_section_names = [
    "Hero", "Feature", "Content", "CTA", "Gallery",
    "Comparison", "Statistics", "Pricing", "Countdown",
    "Timeline", "Contact", "FAQ", "Logo", "Team", "Testimonial"
]


@app.post("/land_summary_menu_generate")
async def land_summary(request: LandPageRequest):
    start = time.time()
    summary = ''
    usr_data = ''
    # ========================
    #    비속어 욕 체크 모듈
    # ========================
    # examine_client = OllamaExamineClient(
    #     model=request.model,
    #     data=request.user_msg
    #     )
    # examine = await examine_client.data_examine()
    # if examine in "비속어":
    #     return "1"

    # ========================
    #      model set 모듈
    # ========================
    model_conf = ModelParam(request.model)
    model_max_token, final_summary_length, max_tokens_per_chunk = model_conf.param_set()
    print(f"request.user_msg : {request.user_msg} \n request.path : {request.path}")

    if request.user_msg != '':
        usr_msg_handle = OllamaUsrMsgClient(usr_msg=request.user_msg, model=request.model)
        usr_data = await usr_msg_handle.usr_msg_process()
    # ========================
    #         PDF 모듈
    # ========================
    if request.path != '':
        pdf_handle = PDFHandle(request.path, request.path2, request.path3)
        pdf_data = pdf_handle.PDF_request()
        # ========================
        #      내용 요약 모듈
        # ========================
        summary_client = OllamaSummaryClient(model=request.model)
        summary = await summary_client.store_chunks_parallel(
            data=pdf_data,
            model_max_token=model_max_token,
            final_summary_length=final_summary_length,
            max_tokens_per_chunk=max_tokens_per_chunk
            )
    # ========================
    #    비속어 욕 체크 모듈
    # ========================
    # examine_client = OllamaExamineClient(
    #     model=request.model,
    #     data=summary
    #     )
    # examine = await examine_client.data_examine()
    # if examine in "비속어":
    #     return "1"
    print(f"usr_data : {len(usr_data)} | {usr_data} \n summary : {len(summary)} | {summary}")
    contents_client = OllamaDataMergeClient(
        model=request.model,
        user_msg=usr_data,
        data=summary
        )
    summary = await contents_client.contents_merge()
    # ========================
    #      메뉴 생성 모듈
    # ========================
    menu_client = OllamaMenuClient(model=request.model)
    section_structure, section_per_context = await menu_client.section_structure_create_logic(summary)

    # 1. 첫 번째 딕셔너리의 값들을 숫자 키 순서대로 추출
    print(f"main section_structure : {section_structure}")
    ordered_new_keys = [section_structure[k] for k in sorted(section_structure, key=int)]
    section_structure_copy = ordered_new_keys.copy()
    ordered_new_keys.insert(0, "Header")
    ordered_new_keys.append("Footer")
    ordered_new_keys[1] = "Hero"

    # 2. 두 번째 딕셔너리의 아이템 목록을 추출 (순서 유지)
    second_items = list(section_per_context.items())
    second_items.insert(0, ('Header', ', '.join(section_structure_copy)))
    second_items.append(('Footer', ', '.join(section_structure_copy)))

    # 3. 순차적으로 매핑하기
    new_dict = {}
    for new_key, (_, value) in zip(ordered_new_keys, second_items):
        new_dict[new_key] = value
    end = time.time()
    t = (end - start)
    print(f"summary : {summary} \n dict : {new_dict}")
    print(f" running time : {t}")
    return summary, new_dict

# section context 데이터와 section structure 데이터를 기반으로
# 섹션 별 태그에 적합한 블럭 추천
# section 별 태그에 값을 입력


class landGen(BaseModel):
    model: str
    block: dict
    section_context: dict

# Example Data


Example = '''
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
async def land_section_generate(request: landGen):
    start = time.time()
    # ==============================================
    #        블록 추천 / 블록 컨텐츠 생성 모듈
    # ==============================================
    content_client = OllamaBlockRecommend(model=request.model)
    content = await content_client.generate_block_content(
        block_list=request.block,
        context=request.section_context
        )

    keyword_client = OllamaKeywordClient(model=request.model)
    keyword = await keyword_client.section_keyword_create_logic(
        request.section_context.keys(),
        request.section_context.values()
        )

    first_key = next(iter(content))  # content의 첫 번째 키 추출
    if first_key in content:
        content[first_key]['keyword'] = keyword  # keyword 추가
    end = time.time()
    t = (end - start)
    print(f" running time : {t}")
    print(f" content : {content}")
    return content

# ===================================================================================
# milvus
# ===================================================================================


@app.post('/openai_faq')
def insert_faq():
    try:
        milvus_handler = MilvusDataHandler()
        # JSON 파일 경로
        faq_data = '/app/블럭데이터.json'
        print(f"FAQ Data Path: {faq_data}")

        # JSON 데이터 로드
        json_data = milvus_handler.load_json(faq_data)
        print(f"Loaded JSON Data: {json_data}")

        # 데이터 전처리
        processed_data = milvus_handler.process_data(json_data)

        # Milvus 컬렉션 생성
        collection = milvus_handler.create_milvus_collection()

        # 데이터 삽입
        milvus_handler.insert_data_to_milvus(collection, processed_data)
        return {"status": "success", "message": "Data inserted into Milvus!"}
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"status": "error", "message": str(e)}


@app.post('/query_search')
def search_db():
    # Milvus에 연결
    # connections.connect(alias="default", host="172.19.0.6", port="19530")

    # 컬렉션 이름
    collection_name = "block_collection"

    # 컬렉션 객체 생성
    collection = Collection(name=collection_name)
    collection.load()

    print(f"컬렉션 '{collection_name}'이 메모리에 로드되었습니다.")
    results = collection.query(
                        expr="",
                        output_fields=[
                            "template_id",
                            "section_type",
                            "emmet_tag",
                            "additional_tags",
                            "embedding",
                            "popularity",
                            "layout_type"
                        ],
                        limit=10
                    )
    for result in results:
        print(result)


# ===================================================================================================
# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize services
openai_config = OpenAIConfig()
openai_service = OpenAIService(openai_config)
batch_handler = BatchRequestHandler(openai_service)

MAX_TOKENS_USR_MSG_PROPOSAL = 500
MAX_TOKENS_SUMMARIZE_TEXT = 1000
MAX_TOKENS_CONTENTS_MERGE = 1500
MAX_TOKENS_CREATE_SECTION_STRUCTURE = 200
MAX_TOKENS_CREATE_SECTION_CONTENTS = 1800
MAX_TOKENS_SELECT_BLOCK = 50
MAX_TOKENS_GENERATE_CONTENTS = 250
MAX_TOKENS_SECTION_KEYWORD_RECOMMEND = 100

@app.post("/batch_completions")
async def batch_completions(requests: List[Completions]):
    try:
        # Convert Pydantic models to dictionaries
        request_dicts = [req.dict() for req in requests]

        # Process batch requests
        response = await batch_handler.process_batch(request_dicts)

        # Check if all requests failed
        if response.get("successful_requests", 0) == 0:
            raise HTTPException(
                status_code=500,
                detail="All batch requests failed"
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) from e


# 동작 설명
# """
# 경우 1: PDF 1개, usr_msg 없음
# 입력: req.pdf_data1 = "퓨처플레이 5000자", req.usr_msg = ""
# 출력:
# pdf_1_proposal: "퓨처플레이는 193개 스타트업에 투자하며 5.7조원 가치를..." (700~1000자)
# final_result: 동일 내용 (API 호출 없이 바로 사용)

# 경우 2: PDF 2개, usr_msg 없음
# 입력: req.pdf_data1 = "PDF1", req.pdf_data2 = "PDF2", req.usr_msg = ""
# 출력:
# pdf_1_proposal: "PDF1 내용..." (700~1000자)
# pdf_2_proposal: "PDF2 내용..." (700~1000자)
# final_result: "PDF1과 PDF2를 합쳐서..." (1000~1500자, 통합 생성)

# 경우 3: PDF 1개, usr_msg 있음
# 입력: req.pdf_data1 = "PDF1", req.usr_msg = "삼쩜삼은 세무 서비스야"
# 출력:
# usr_msg: "삼쩜삼은 세무를 간단히..." (500~700자)
# pdf_1_proposal: "PDF1 내용..." (700~1000자)
# final_result: "삼쩜삼은 세무를 편리하게 하고, PDF1에서 영감을 받아..." (1000~1500자)

# 경우 4: usr_msg만 있음, PDF 없음
# 입력: req.pdf_data1 = "", req.usr_msg = "삼쩜삼은 세무 서비스야"
# 출력:
# usr_msg: "삼쩜삼은 세무를 간단히..." (500~700자)
# final_result: 동일 내용 (API 호출 없이 바로 사용)"""
async def inputDataProcess(req, req_idx):
    results = []
    try:
        # PDF 데이터 준비
        pdf_data_list = [req.pdf_data1 or "", req.pdf_data2 or "", req.pdf_data3 or ""]
        pdf_data_list = [pdf for pdf in pdf_data_list if pdf.strip()]
        usr_msg = req.usr_msg if hasattr(req, 'usr_msg') else ""

        # 유저 Proposal 생성 (PDF와 독립)
        usr_client = OpenAIUsrMsgClient(usr_msg, batch_handler)
        usr_task = usr_client.usr_msg_proposal() if usr_msg else None

        # PDF별 Proposal 생성 병렬 처리
        proposal_clients = [OpenAIProposalClient(pdf, batch_handler) for pdf in pdf_data_list]
        proposal_tasks = [client.generate_proposal() for client in proposal_clients]

        # 유저 Proposal과 PDF Proposal 동시 처리
        tasks = [task for task in [usr_task] + proposal_tasks if task]  # None 제외
        if not tasks:  # usr_msg와 PDF 모두 없는 경우
            return [{"type": "error", "result": {"success": False, "data": None, "error": "입력 데이터 없음"}}]
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        usr_result = results_list[0] if usr_msg else None
        pdf_proposal_results = results_list[1:] if usr_msg else results_list

        # 결과 추출
        usr_proposal = usr_result['generations'][0][0]['text'] if isinstance(usr_result, dict) and "error" not in usr_result else ""
        pdf_proposals = [
            pr['generations'][0][0]['text'] if isinstance(pr, dict) and "error" not in pr else ""
            for pr in pdf_proposal_results
        ]
        pdf_proposals = [p for p in pdf_proposals if p.strip()]

        # 최종 Proposal 처리
        if usr_proposal or pdf_proposals:
            if usr_proposal and not pdf_proposals:  # usr_msg만 있는 경우
                comp_result = {
                    "generations": [[{"text": usr_proposal}]]  # 유저 Proposal 그대로 사용
                }
            elif len(pdf_proposals) == 1 and not usr_proposal:  # PDF 1개만 있는 경우
                comp_result = {
                    "generations": [[{"text": pdf_proposals[0]}]]  # PDF Proposal 그대로 사용
                }
            else:  # PDF 2개 이상이거나 usr_proposal과 PDF가 함께 있는 경우
                proposal_client = OpenAIProposalClient("", batch_handler)  # 빈 pdf_content로 인스턴스 생성
                if len(pdf_proposals) >= 2 and not usr_proposal:  # PDF 2개 이상, usr_msg 없음
                    comp_result = await proposal_client.consolidate_proposals(pdf_proposals)
                else:  # usr_proposal과 PDF가 함께 있거나 PDF 1개 + usr_proposal
                    comp_client = OpenAIComprehensiveProposalClient(usr_proposal, pdf_proposals, batch_handler)
                    comp_result = await comp_client.generate_comprehensive_proposal()

            # 종합 Proposal 결과 추가
            if "error" not in comp_result:
                results.append({
                    "type": "final_result",
                    "result": {
                        "success": True,
                        "data": comp_result,
                        "error": None,
                        "error_details": None
                    }
                })
            else:
                results.append({
                    "type": "final_result",
                    "result": {
                        "success": False,
                        "data": None,
                        "error": comp_result["error"],
                        "error_details": None
                    }
                })

        # 개별 결과 추가
        if usr_proposal:
            results.append({
                "type": "usr_msg",
                "result": {"success": isinstance(usr_result, dict) and "error" not in usr_result, "data": usr_result}
            })
        for i, pr in enumerate(pdf_proposal_results):
            results.append({
                "type": f"pdf_{i+1}_proposal",
                "result": {"success": isinstance(pr, dict) and "error" not in pr, "data": pr}
            })

        return results
    except Exception as e:
        logger.error(f"요청 {req_idx} 처리 오류: {str(e)}")
        return [{"type": "error", "result": {"success": False, "data": None, "error": str(e)}}]
        
# async def inputDataProcess(req, req_idx):
#     results = []
#     try:
#         # PDF 데이터 준비
#         pdf_data_list = [req.pdf_data1 or "", req.pdf_data2 or "", req.pdf_data3 or ""]
#         pdf_data_list = [pdf for pdf in pdf_data_list if pdf.strip()]
#         usr_msg = req.usr_msg if hasattr(req, 'usr_msg') else ""

#         # 유저 Proposal 생성 (PDF와 독립)
#         usr_client = OpenAIUsrMsgClient(usr_msg, batch_handler)
#         usr_task = usr_client.usr_msg_proposal()

#         # PDF별 요약 병렬 처리
#         summary_clients = [OpenAIPDFSummaryClient(pdf, batch_handler) for pdf in pdf_data_list]
#         summary_tasks = [client.summarize_pdf(pdf) for client, pdf in zip(summary_clients, pdf_data_list)]
        
#         # 유저 Proposal과 PDF 요약 동시 처리
#         usr_result, *summary_results = await asyncio.gather(usr_task, *summary_tasks, return_exceptions=True)

#         # PDF별 Proposal 생성 병렬 처리
#         proposal_clients = [
#             OpenAIProposalClient(
#                 sr['generations'][0][0]['text'] if isinstance(sr, dict) and "error" not in sr else "",
#                 batch_handler
#             )
#             for sr in summary_results
#         ]
#         proposal_tasks = [client.generate_proposal() for client in proposal_clients]
#         pdf_proposal_results = await asyncio.gather(*proposal_tasks, return_exceptions=True)

#         # 결과 추출
#         usr_proposal = usr_result['generations'][0][0]['text'] if isinstance(usr_result, dict) and "error" not in usr_result else ""
#         pdf_proposals = [
#             pr['generations'][0][0]['text'] if isinstance(pr, dict) and "error" not in pr else ""
#             for pr in pdf_proposal_results
#         ]
#         pdf_proposals = [p for p in pdf_proposals if p.strip()]

#         # 종합 Proposal 생성
#         if usr_proposal or pdf_proposals:
#             comp_client = OpenAIComprehensiveProposalClient(usr_proposal, pdf_proposals, batch_handler)
#             comp_result = await comp_client.generate_comprehensive_proposal()
#             if "error" not in comp_result:
#                 results.append({
#                     "type": "final_result",
#                     "result": {
#                         "success": True,
#                         "data": comp_result,
#                         "error": None,
#                         "error_details": None
#                     }
#                 })
#             else:
#                 results.append({
#                     "type": "final_result",
#                     "result": {
#                         "success": False,
#                         "data": None,
#                         "error": comp_result["error"],
#                         "error_details": None
#                     }
#                 })

#         # 개별 결과 추가
#         results.append({
#             "type": "usr_msg",
#             "result": {"success": isinstance(usr_result, dict) and "error" not in usr_result, "data": usr_result}
#         })
#         for i, (sr, pr) in enumerate(zip(summary_results, pdf_proposal_results)):
#             results.append({
#                 "type": f"pdf_{i+1}_summary",
#                 "result": {"success": isinstance(sr, dict) and "error" not in sr, "data": sr}
#             })
#             results.append({
#                 "type": f"pdf_{i+1}_proposal",
#                 "result": {"success": isinstance(pr, dict) and "error" not in pr, "data": pr}
#             })

#         return results
#     except Exception as e:
#         logger.error(f"요청 {req_idx} 처리 오류: {str(e)}")
#         return [{"type": "error", "result": {"success": False, "data": None, "error": str(e)}}]

# async def inputDataProcess(req, req_idx):
#     results = []
#     try:
#         # PDF 데이터와 usr_msg 준비
#         pdf_data = req.pdf_data1 + (req.pdf_data2 or "") + (req.pdf_data3 or "") if req.pdf_data1 else ""
#         usr_msg = req.usr_msg if hasattr(req, 'usr_msg') else ""

#         # 클라이언트 초기화
#         summary_client = OpenAIPDFSummaryClient(pdf_data, batch_handler) if pdf_data else None
#         usr_msg_client = OpenAIUsrMsgClient(usr_msg, batch_handler) if usr_msg else None

#         # 병렬 처리
#         summary_result, usr_result = await asyncio.gather(
#             summary_client.summarize_pdf(content=pdf_data) if summary_client else asyncio.sleep(0),
#             usr_msg_client.usr_msg_proposal() if usr_msg_client else asyncio.sleep(0)
#         )
#         print(f"summary_result, usr_result : {summary_result} \n {usr_result}")
#         # 사용자 입력 결과 추가 ("usr_msg_argument")
#         if usr_result:
#             if "error" not in usr_result:
#                 results.append({
#                     "type": "usr_msg_argument",
#                     "result": {
#                         "success": True,
#                         "data": usr_result,
#                         "error": None,
#                         "error_details": None
#                     }
#                 })
#             else:
#                 results.append({
#                     "type": "usr_msg_argument",
#                     "result": {
#                         "success": False,
#                         "data": None,
#                         "error": usr_result["error"],
#                         "error_details": str(usr_result.get("error_details", ""))
#                     }
#                 })

#         # PDF 요약 결과 추가 ("pdf_summary")
#         if summary_result:
#             if "error" not in summary_result:
#                 results.append({
#                     "type": "pdf_summary",
#                     "result": {
#                         "success": True,
#                         "data": summary_result,
#                         "error": None,
#                         "error_details": None
#                     }
#                 })
#             else:
#                 results.append({
#                     "type": "pdf_summary",
#                     "result": {
#                         "success": False,
#                         "data": None,
#                         "error": summary_result["error"],
#                         "error_details": str(summary_result.get("error_details", ""))
#                     }
#                 })

#         # final_result 생성
#         if usr_result and "error" not in usr_result and not summary_result:  # usr_msg만 있는 경우
#             results.append({
#                 "type": "final_result",
#                 "result": {
#                     "success": True,
#                     "data": usr_result,  # usr_msg만 있을 때는 그대로 사용
#                     "error": None,
#                     "error_details": None
#                 }
#             })
#         elif summary_result and "error" not in summary_result and not usr_result:  # pdf만 있는 경우
#             results.append({
#                 "type": "final_result",
#                 "result": {
#                     "success": True,
#                     "data": summary_result,  # pdf만 있을 때는 그대로 사용
#                     "error": None,
#                     "error_details": None
#                 }
#             })
#         elif (summary_result and "error" not in summary_result) and (usr_result and "error" not in usr_result):  # 둘 다 있는 경우
#             summary_text = summary_result['generations'][0][0]['text']
#             usr_text = usr_result['generations'][0][0]['text']
#             merge_client = OpenAIDataMergeClient(usr_text, summary_text, batch_handler)
#             merge_result = await merge_client.contents_merge(max_tokens=800)
#             if merge_result and "error" not in merge_result:
#                 re_text = merge_result['generations'][0][0]['text'].replace("\n", " ")
#                 merge_result['generations'][0][0]['text'] = re_text
#                 results.append({
#                     "type": "final_result",
#                     "result": {
#                         "success": True,
#                         "data": merge_result,
#                         "error": None,
#                         "error_details": None
#                     }
#                 })
#             else:
#                 results.append({
#                     "type": "final_result",
#                     "result": {
#                         "success": False,
#                         "data": None,
#                         "error": merge_result.get("error", "Merge failed") if merge_result else "Merge result is None",
#                         "error_details": str(merge_result) if merge_result else "No merge result returned"
#                     }
#                 })

#         return results
#     except Exception as e:
#         logger.error(f"요청 {req_idx} 처리 오류: {str(e)}", exc_info=True)
#         return [{
#             "type": "error",
#             "result": {
#                 "success": False,
#                 "data": None,
#                 "error": str(e),
#                 "error_details": str(e.__traceback__)
#             }
#         }]

# 상위 호출 코드
@app.post("/api/input_data_process")
async def openai_input_data_process(requests: List[Completions]):
    try:
        start = time.time()
        tasks = [inputDataProcess(req, idx) for idx, req in enumerate(requests)]
        processed_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for result in processed_results:
            if isinstance(result, Exception):
                results.append({
                    "type": "error",
                    "result": {
                        "success": False,
                        "data": None,
                        "error": str(result),
                        "error_details": str(result.__traceback__)
                    }
                })
            else:
                results.extend(result)

        end = time.time()
        return {
            "timestamp": end - start,
            "total_requests": len(requests),
            "successful_requests": sum(1 for r in results if r["result"]["success"]),
            "failed_requests": sum(1 for r in results if not r["result"]["success"]),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# @app.post("/api/input_data_process")
# async def openai_input_data_process(requests: List[Completions]):
#     try:
#         start = time.time()
#         results = []

#         for req in requests:
#             usr_msg_result = None  # 초기값을 None으로 변경
#             summary_result = None  # 초기값을 None으로 변경
#             summary = None
#             usr_msg = None

#             # usr_msg 처리
#             if req.usr_msg:
#                 usr_msg_client = OpenAIUsrMsgClient(req.usr_msg, batch_handler)
#                 usr_msg_result = await usr_msg_client.usr_msg_proposal(max_tokens=MAX_TOKENS_USR_MSG_PROPOSAL) 
#                 results.append({"type": "usr_msg_argument", "result": usr_msg_result})

#             # PDF 요약 처리
#             if req.pdf_data1:
#                 try:
#                     pdf_data = req.pdf_data1 + (req.pdf_data2 or "") + (req.pdf_data3 or "")
#                     summary_client = OpenAIPDFSummaryClient(pdf_data, batch_handler)
#                     # NOTE : 
#                     summary_result = await summary_client.summarize_chunked_texts_with_CoD(pdf_data, 2000, 500)
#                     # summary_result = await summary_client.summarize_text(pdf_data, max_tokens=MAX_TOKENS_SUMMARIZE_TEXT)
#                     results.append({"type": "pdf_summary", "result": summary_result})
#                 except Exception as e:
#                     print(f"Error in PDFHandle: {str(e)}")
#                     results.append({"type": "pdf_summary", "error": str(e)})
#                     continue
#             try:
#                 if usr_msg_result and summary_result:
#                     # usr_msg_result에서 텍스트 추출
#                     if hasattr(usr_msg_result.data, 'generations'):
#                         usr_msg = usr_msg_result.data['generations'][0][0]['text']
#                     else:
#                         usr_msg = str(usr_msg_result.data)

#                     # summary_result에서 텍스트 추출
#                     if isinstance(summary_result.data, dict) and 'generations' in summary_result.data:
#                         summary = summary_result.data['generations'][0][0]['text']
#                     else:
#                         summary = str(summary_result.data)
#                     merge_client = OpenAIDataMergeClient(usr_msg, summary, batch_handler)
#                     merge_result = await merge_client.contents_merge(max_tokens=MAX_TOKENS_CONTENTS_MERGE)
#                     re_text = merge_result.data['generations'][0][0]['text']
#                     re_text = re_text.replace("\n", " ")
#                     merge_result.data['generations'][0][0]['text'] = re_text
#                     results.append({"type": "final_result", "result": merge_result})

#                 elif usr_msg_result and usr_msg_result.data['generations'][0][0]['text']:
#                     results.append({"type": "final_result", "result": usr_msg_result})
#                 elif summary_result and summary_result.data['generations'][0][0]['text']:
#                     temp_results = await summary_client.generate_proposal(summary_result.data['generations'][0][0]['text'])
#                     results.append({"type": "final_result", "result": temp_results})
#             except Exception as e:
#                 print(f"merge process error: {e}")
#                 results.append({"type": "final_result", "error": str(e)})
        
#         end = time.time()
#         processing_time = end - start
#         response = {
#             "timestamp": processing_time,
#             "total_requests": len(requests),
#             "successful_requests": sum(1 for r in results if "error" not in r),
#             "failed_requests": sum(1 for r in results if "error" in r),
#             "results": results
#         }
#         return response

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e)) from e
    
# NOTE 250219: API 이름 바꾸기 논의
# @app.post("/api/section_select_n_content_generate")
@app.post("/api/section_select")
async def openai_section_select(requests: List[Completions]):
    """Landing page section generation API"""
    try:
        start = time.time()
        # logger.info(f"Received section generation request: {requests}")
        
        generator = OpenAISectionGenerator(batch_handler)

        results = await generator.generate_landing_page(requests, max_tokens=MAX_TOKENS_CREATE_SECTION_STRUCTURE)
        
        end = time.time()
        processing_time = end - start
        # logger.info(f"Processing time: {processing_time} seconds")s
        
        response = {
            "timestamp": processing_time,
            "total_requests": len(requests),
            "successful_requests": sum(1 for r in results if all(r.values())),
            "failed_requests": sum(1 for r in results if not all(r.values())),
            "results": results
        }

        return response

    except Exception as e:
        logger.error(f"Error in section generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/block_select")
async def openai_block_select(requests: List[Completions]):
    try:
        start = time.time()
        blockselect_client = OpenAIBlockSelector(batch_handler=batch_handler)
        BATCH_SIZE = 10 
        batched_requests = [requests[i:i + BATCH_SIZE] for i in range(0, len(requests), BATCH_SIZE)]
        
        final_results = []
        tasks = []
        
        # 각 배치에 대한 비동기
        for batch in batched_requests:
            block_lists = [req.block for req in batch]
            contexts = [req.section_context for req in batch]
            
            # 비동기
            task = blockselect_client.select_block_batch(contexts, block_lists, max_tokens=MAX_TOKENS_SELECT_BLOCK)
            tasks.append(task)
        
        # 모든 배치 작업을 병렬로 실행
        batch_results = await asyncio.gather(*tasks)

        for result in batch_results:
            final_results.append(result)

        end = time.time()
        processing_time = end - start

        # 성공/실패 요청 수 계산
        flat_results = [item for sublist in final_results for item in sublist]

        response = {
            "timestamp": processing_time,
            "total_requests": len(requests),
            "successful_requests": len(flat_results),
            "failed_requests": len(requests) - len(flat_results),
            "results": final_results
        }
        
        return response

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) from e


# @app.post("/api/block_select")
# async def openai_block_select(requests: List[Completions]):
#     try:
#         start = time.time()
#         blockselect_client = OpenAIBlockSelector(batch_handler=batch_handler)

#         # logger.debug(f"Received requests: {requests}")

#         block_lists = [req.block for req in requests]
#         # logger.debug(f"Extracted block_lists: {block_lists}")

#         contexts = [req.section_context for req in requests]
#         # logger.debug(f"Extracted contexts: {contexts}")

#         # logger.debug("Starting generate_block_content_batch")
#         final_results = []        
#         select_block_result = await blockselect_client.select_block_batch(contexts, block_lists, max_tokens=MAX_TOKENS_SELECT_BLOCK)
#         # logger.debug(f"Results from generate_block_content_batch: {results}")
#         final_results.append(select_block_result)
#         end = time.time()
#         processing_time = end - start
#         # logger.info(f"Processing time: {processing_time} seconds")
        
#         # NOTE 250220 : 잠시 결과물을 위해서 batch 처리 제거. 
#         #               그래서 successful_requests, failed_requests를 빼놓은 상태. 추후에 batch 넣으면서 살릴 것
#         response = {
#             "timestamp": processing_time,
#             "total_requests": len(requests),
#             # "successful_requests": sum(1 for r in results if not r.error),
#             # "failed_requests": sum(1 for r in results if not r.error),
#             "results": final_results
#         }
        
#         return response

#     except Exception as e:
#         logger.error(f"Error occurred: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         ) from e

# 이거 랜딩페이지 만들 수 있게 작업해야함.
# FastAPI 엔드포인트

@app.post("/api/block_content_generate")
async def openai_block_content_generate(requests: List[Completions]):
    try:
        start = time.time()
        blockcontentclient = OpenAIBlockContentGenerator(batch_handler=batch_handler)
        keywordclient = OpenAIKeywordClient(batch_handler=batch_handler)

        async def content_batch_process(req, blockcontentclient, keywordclient):
            try:
                # 각 요청 내에서도 content와 keyword 생성을 병렬로 처리
                content_task = blockcontentclient.generate_content(req.tag_length, req.section_context, max_tokens=1000)
                keyword_task = keywordclient.section_keyword_create_logic(context=next(iter(req.section_context.values())), max_tokens=MAX_TOKENS_SECTION_KEYWORD_RECOMMEND)

                # 두 작업을 동시에 실행
                content_result, keyword_result = await asyncio.gather(content_task, keyword_task)

                return {
                    "content": content_result,
                    "keywords": keyword_result
                }
            except Exception as e:
                return e  # 예외를 반환하여 상위 레벨에서 처리

        # 각 요청에 대한 처리 작업 생성
        tasks = [content_batch_process(req, blockcontentclient, keywordclient) for req in requests]
        print("What is in the tasks : ", tasks)
        
        # 모든 태스크를 병렬로 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)
                
        end = time.time()
        processing_time = end - start
        response = {
            "timestamp": processing_time,
            "total_requests": len(requests),
            "successful_requests": sum(1 for r in processed_results if "error" not in r),
            "failed_requests": sum(1 for r in processed_results if "error" in r),
            "results": processed_results
        }
        return response
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/api/block_content_generate")
# async def openai_block_content_generate(requests: List[Completions]):
#     try:
#         start = time.time()
#         blockcontentclient = OpenAIBlockContentGenerator(batch_handler=batch_handler)
#         keywordclient = OpenAIKeywordClient(batch_handler=batch_handler)
#         results = []
#         for req in requests:            
#             content_result = await blockcontentclient.generate_content(req.tag_length, req.section_context, max_tokens=1000)
#             keyword_result = await keywordclient.section_keyword_create_logic(context=next(iter(req.section_context.values())), max_tokens=MAX_TOKENS_SECTION_KEYWORD_RECOMMEND)
#             combined_result = {
#                 "content": content_result,
#                 "keywords": keyword_result
#                 }
#             results.append(combined_result)
#         end = time.time()
#         processing_time = end - start
#         response = {
#             "timestamp": processing_time,
#             "total_requests": len(requests),
#             "successful_requests": sum(1 for r in results if "error" not in r),
#             "failed_requests": sum(1 for r in results if "error" in r),
#             "results": results
#         }
#         return response
#     except ValueError as ve:
#         logger.error(f"Validation error: {str(ve)}")
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/block_content_generate")
# async def openai_block_content_generate(requests: List[Completions]):
#     try:
#         start = time.time()
#         blockcontentclient = OpenAIBlockContentGenerator(batch_handler=batch_handler)
#         keywordclient = OpenAIKeywordClient(batch_handler=batch_handler)
#         async def content_batch_process(req, blockcontentclient, keywordclient):
#             try:
#                 # 각 요청 내에서도 content와 keyword 생성을 병렬로 처리
#                 content_task = blockcontentclient.generate_content(req.tag_length, req.section_context, max_tokens=1000)
#                 keyword_task = keywordclient.section_keyword_create_logic(context=next(iter(req.section_context.values())), max_tokens=MAX_TOKENS_SECTION_KEYWORD_RECOMMEND)
                
#                 # 두 작업을 동시에 실행
#                 content_result, keyword_result = await asyncio.gather(content_task, keyword_task)
                
#                 return {
#                     "content": content_result,
#                     "keywords": keyword_result
#                 }
#             except Exception as e:
#                 return e  # 예외를 반환하여 상위 레벨에서 처리
#         results = []
#         tasks = [content_batch_process(req, blockcontentclient, keywordclient) for req in requests]
        
#         # 모든 태스크를 병렬로 실행
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         # 예외 처리
#         processed_results = []
#         for result in results:
#             if isinstance(result, Exception):
#                 processed_results.append({"error": str(result)})
#             else:
#                 processed_results.append(result)
#         end = time.time()
#         processing_time = end - start
#         response = {
#             "timestamp": processing_time,
#             "total_requests": len(requests),
#             "successful_requests": sum(1 for r in results if "error" not in r),
#             "failed_requests": sum(1 for r in results if "error" in r),
#             "results": results
#         }
#         return response
#     except ValueError as ve:
#         logger.error(f"Validation error: {str(ve)}")
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

