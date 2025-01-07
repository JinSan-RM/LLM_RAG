1. LangChain Folder에 들어온 상태에서 docker compose build --no-cache
   (※ 모델 디렉토리를 langchain 경로에 놓고 build 진행)

2. docker compose up

3. 터미널을 새로 켜서 아래 명령어를 실행. {"status":"ok"}를 확인
   curl http://localhost:8000/healthcheck

4. 각 이미지별로 debug, info log message 확인 후 점검



--------
kubernetes
kubectl apply -f kubernetes/ollama/deployment.yaml #쿠버네티스 설치
kubectl apply -f kubernetes/ollama/service.yaml

kubectl get pods --all-namespaces 쿠버네티스 관리 pods 전체 서치

kubectl exec -it ollama-deployment-b796b4bf5-hj5q2 -n default -- /bin/bash
--------

LLM - llama를 이용한 생성 방식 (한글 영어 상관 없음)
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{
           "input_text": "llama3.2 에 대해 알려줘",
           "model": "llama3.2"
         }'

LLM llama milvus 상태 체크
curl http://localhost:8000/healthcheck

LLM llama milvus의 db 생성
curl -X POST http://localhost:8000/api/db_create \
        -H "Content-Type: application/json" \
        -d '{
                "name": "ko_std_industry_collection"
            }'


http://localhost:8000/api/question_search?collection=ko_std_industry_collection&quert_text=트렌드알려줘

milvus 연결 체크
http://localhost:8000/api/db_connection

milvus insert 

curl -X POST http://localhost:8000/api/insert \
        -H "COntent-Type: application/json" \
        -d '{
                {
                    "question": "question 질문",
                    "answer": "answer for question 질문에 대한 답변",
                    "metadata": {
                        "First_Category": "도매 및 상품 중개업",
                        "Second_Category": "도매 및 상품 중개업",
                        "Third_Category": "상품 종합 도매업",
                        "Fourth_Category": "상품 종합 도매업",
                        "Fifth_Category": "상품 종합 도매업",
                        "Menu": "#main-content",
                        "est_date": "20160503",
                        "corp_name": "위인코리아_1",
                        "question_template": "business_question_template"
                        }
                }
            }'



--------


### Milvus DB 생성 전 해야할 것들

**(※ 아래를 주석해놓지 않으면 실행이 안돼서, 통신까지 가지 못함)**

1. utils 디렉토리 > __init__.py 에서 아래 줄 주석
from .milvus_collection import CONTENTS_COLLECTION_MILVUS_STD

2. utils 디렉토리 > milvus_collection.py 에서 전체 주석
from pymilvus import connections, Collection, utility
CONTENTS_COLLECTION_MILVUS_STD = collection = Collection("ko_std_industry_collection")

3. docker compose up 한 후에 postman에서 아래 실행
(GET) http://localhost:8000/healthcheck
-> {status : "OK"} 확인

4. postman에서 아래 실행
(POST) http://localhost:8000/api/db_create  
Body {
    "name": "ko_std_industry_collection"
}
-> Milvus DB 생성

5. 정상적으로 생성 됐을 경우, 1, 2에서 주석했던 것들을 풀어서 재실행해주면 완료

6. PDF to Menu 실행 코드
(POST) http://localhost:8001/generate_menu?path=[cdn 주소_1]&path2=[cdn 주소_2]&path3=[cdn 주소_3]


--------

### Docker Cash 지우고 build

docker container prune

docker image prune -a

docker volume prune -a

docker compose build --no-cache --progress=plain

--------
사용 방법. (테스트할 때 사용용)
# 섹션 구조와 summary, 섹션별 context 내용 생성 API
curl -X POST http://192.168.0.42:8001/land_section_generate \
-H "Content-Type: application/json" \
-d "{"path":"cdn경로에 있는 pdf 파일 파일경로 입력", "model": "AI model 명"}

ex. -d"{"path" : "https://cdn.zaemit.com/weven_data/app_weven/ai/PDF/회사소개서_KG이니시스.pdf", "model" : "EEVE"}"

기대값:
"KG이니시스는 1998년 설립된 전자결제 전문 기업으로, 시장점유율 1위 PG사입니다. 230명의 전문가로 구성된 경쟁력 있는 팀을 보유하고 있으며, 업계 최초 충전식 전자화폐 출시 및 국내 최초 모바일 페이먼트 서비스 설립 등 다양한 이정표를 세웠습니다. KG이니시스는 안전한 거래와 결제 솔루션을 제공하며, 통합간편결제서비스를 통해 가맹점의 결제창에 다양한 간편결제를 제공합니다. 또한, 업계 최다 기술특허를 보유하고 있으며, PCI-DSS 인증을 8년 연속으로 받았습니다.\r\n\r\nKG이니시스는 온라인쇼핑몰 거래액에서 꾸준한 성장을 이루었으며, 2019년에는 연간거래금액이 23.7조원에 달했습니다. 또한, 다양한 사업영역을 통해 전자결제 및 결제 솔루션을 제공하고 있습니다. 예를 들어, VAN(Value Added Network) 서비스는 O2O 구분 없이 다양한 서비스를 제공하며 안정적인 VAN 서비스 제공을 위해 자체 솔루션으로 SI 가맹점 서비스를 제공합니다.\r\n\r\nKG이니시스의 브랜드파워는 전자결제 분야에서 대한민국의 대표 브랜드로 자리매김하고 있으며, 업계 최다 기술특허 보유와 PCI-DSS 인증 획득을 통해 글로벌 보안 기준을 준수합니다. 우수한 재무 안정성을 바탕으로 한 경쟁사 대비 월등한 유동성 보유 및 안정적인 이윤 창출이 가능하며, 다양한 부가서비스를 제공합니다.\r\n\r\nKG이니시스는 INIPAY 전자결제 서비스, INI VAN/테이블오더 서비스, 이니렌탈서비스 등 다양한 사업영역을 통해 고객에게 안전하고 편리한 결제 솔루션을 제공하고 있습니다. 또한, 광고플랫폼 및 부가서비스를 통해 종합적인 전자결제 서비스를 제공합니다.\r\n\r\n요약하자면, KG이니시스는 시장점유율 1위 PG사로 안전한 거래와 결제 솔루션을 제공하며 다양한 사업영역을 통해 고객에게 종합적인 전자결제 서비스를 제공하는 선도 기업입니다. KG이니시스는 시장점유율 1위 전자결제 전문 기업으로, 안전한 거래 및 결제 솔루션을 제공합니다. 다양한 사업영역을 통해 통합간편결제서비스, INIPAY 오픈웹서비스, 모바일 서비스, ARS 서비스, VA N(Value Added Network) 서비스, 키오스크 및 테이블 오더 서비스를 제공합니다. KG이니시스는 신용카드, 체크카드, 멤버십 카드 등 다양한 결제수단을 지원하며, 현금IC 거래, 간편결제/해외카드 서비스도 제공합니다. 또한, 통합간편결제서비스로 가맹점의 브랜드에 맞는 자체 간편결제 서비스를 구축할 수 있습니다.\r\n\r\nKG이니시스는 키오스크 및 테이블 오더 서비스를 제공하여 비대면 결제를 가능하게 하며, 종교재단 및 비영리 단체에도 카드/간편 결제 기부를 제공합니다. 산업별 맞춤형 솔루션을 제공하며 원격 지원을 통해 시스템을 업데이트하고 다양한 결제수단을 확장할 수 있습니다. INIPAY PC 및 모바일 결제창 광고 플랫폼과 가맹점 관리자사이트를 통한 B2B 플랫폼 광고를 제공하여 광고 서비스를 제공합니다.\r\n\r\nKG이니시스는 실시간 번역 서비스, 카드 본인확인 서비스 등 부가서비스를 제공하며, 일본 결제를 위한 TOtal 솔루션을 통해 한국 및 일본 소비자를 대상으로 한 결제 솔루션을 제공합니다. 또한, 가맹점의 브랜드에 맞는 자체 간편결제 서비스를 구축할 수 있는 WPAY(더블유페이) 서비스와 최대 60개월까지 월분할 납부가 가능한 이니렌탈 구독 결제도 제공합니다.\r\n\r\n요약하자면, KG이니시스는 시장점유율 1위 전자결제 전문 기업으로 안전한 거래 및 결제 솔루션을 제공하며 다양한 사업영역을 통해 고객에게 종합적인 전자결제 서비스를 제공합니다. KG이니시스는 시장점유율 1위 전자결제 전문 기업으로, 안전한 거래 및 결제 솔루션을 제공합니다. 다양한 사업영역을 통해 통합간편결제서비스, INIPAY 오픈웹서비스, 모바일 서비스, ARS 서비스, VA N(Value Added Network) 서비스, 키오스크 및 테이블 오더 서비스를 제공합니다. KG이니시스는 신용카드, 체크카드, 멤버십 카드 등 다양한 결제수단을 지원하며 현금IC 거래, 간편결제/해외카드 서비스도 제공합니다.\r\n\r\nKG이니시스는 키오스크 및 테이블 오더 서비스를 제공하여 비대면 결제를 가능하게 하며, 종교재단 및 비영리 단체에도 카드/간편 결제 기부 기능을 제공합니다. 산업별 맞춤형 솔루션을 제공하며 원격 지원을 통해 시스템을 업데이트하고 다양한 결제수단을 확장할 수 있습니다. INIPAY PC 및 모바일 결제창 광고 플랫폼과 가맹점 관리자사이트를 통한 B2B 플랫폼 광고를 제공하여 광고 서비스를 제공합니다.\r\n\r\nKG이니시스는 실시간 번역 서비스, 카드 본인확인 서비스 등 부가서비스를 제공하며 일본 결제용 TOtal 솔루션을 통해 한국 및 일본 소비자 대상의 결제 솔루션을 제공합니다. 또한, 가맹점 브랜드에 맞는 자체 간편결제 서비스 구축이 가능한 WPAY(더블유페이) 서비스와 최대 60개월까지 월분할 납부가 가능한 이니렌탈 구독 결제도 제공합니다.\r\n\r\n요약하자면, KG이니시스는 시장점유율 1위 전자결제 전문 기업으로 안전한 거래 및 결제 솔루션을 제공하며 다양한 사업영역을 통해 고객에게 종합적인 전자결제 서비스를 제공합니다. KG이니시스는 시장점유율 1위 전자결제 전문 기업으로, 안전한 거래 및 결제 솔루션을 제공합니다. 다양한 사업영역을 통해 통합간편결제서비스, INIPAY 오픈웹서비스, 모바일 서비스, ARS 서비스, VA N(Value Added Network) 서비스, 키오스크 및 테이블 오더 서비스를 제공합니다. KG이니시스는 신용카드, 체크카드, 멤버십 카드 등 다양한 결제수단을 지원하며 현금IC 거래, 간편결제/해외카드 서비스도 제공합니다.\r\n\r\nKG이니시스는 키오스크 및 테이블 오더 서비스를 제공하여 비대면 결제를 가능하게 하며, 종교재단 및 비영리 단체에도 카드/간편 결제 기부 기능을 제공합니다. 산업별 맞춤형 솔루션을 제공하며 원격 지원을 통해 시스템을 업데이트하고 다양한 결제수단을 확장할 수 있습니다. INIPAY PC 및 모바일 결제창 광고 플랫폼과 가맹점 관리자사이트를 통한 B2B 플랫폼 광고를 제공하여 광고 서비스를 제공합니다.\r\n\r\nKG이니시스는 실시",
    {
        "1": "Navbars",
        "2": "Hero Header Sections",
        "3": "Feature Sections",
        "4": "Content Sections",
        "5": "Testimonial Sections",
        "6": "CTA Sections",
        "7": "Pricing Sections",
        "8": "Contact Sections",
        "9": "Footers"
    },
    {
        "Navbars": "KG이니시스는 시장점유율 1위 전자결제 전문 기업으로 안전한 거래 및 결제 솔루션을 제공합니다. 다양한 사업영역을 통해 고객에게 종합적인 전자결제 서비스를 제공하는 선도 기업입니다.",
        "Herp Header Sections": "KG이니시스의 브랜드파워는 전자결제 분야에서 대한민국의 대표 브랜드로 자리매김하고 있으며, 업계 최다 기술특허 보유와 PCI-DSS 인증 획득을 통해 글로벌 보안 기준을 준수합니다. 우수한 재무 안정성을 바탕으로 한 경쟁사 대비 월등한 유동성 보유 및 안정적인 이윤 창출이 가능하며 다양한 부가서비스를 제공합니다.",
        "Feature Sections": "KG이니시스는 INIPAY 전자결제 서비스, INI VA N/테이블오더 서비스, 이니렌탈서비스 등 다양한 사업영역을 통해 고객에게 안전하고 편리한 결제 솔루션을 제공합니다. 또한, 광고플랫폼 및 부가서비스를 통해 종합적인 전자결제 서비스를 제공합니다.",
        "Content Sections": "KG이니시스는 키오스크 및 테이블 오더 서비스를 제공하여 비대면 결제를 가능하게 하며, 종교재단 및 비영리 단체에도 카드/간편 결제 기부 기능을 제공합니다. 산업별 맞춤형 솔루션을 제공하며 원격 지원을 통해 시스템을 업데이트하고 다양한 결제수단을 확장할 수 있습니다.",
        "Testimonial Sections": "KG이니시스는 실시간 번역 서비스, 카드 본인확인 서비스 등 부가서비스를 제공하며 일본 결제용 TOtal 솔루션을 통해 한국 및 일본 소비자 대상의 결제 솔루션을 제공합니다. 또한, 가맹점 브랜드에 맞는 자체 간편결제 서비스 구축이 가능한 WPAY(더블유페이) 서비스와 최대 60개월까지 월분할 납부가 가능한 이니렌탈 구독 결제도 제공합니다.",
        "CTA Sections": "KG이니시스는 시장점유율 1위 전자결제 전문 기업으로 안전한 거래 및 결제 솔루션을 제공하며 다양한 사업영역을 통해 고객에게 종합적인 전자결제 서비스를 제공합니다. KG이니시스는 시장점유율 1위 전자결제 전문 기업으로, 다양한 사업영역을 통해 통합간편결제서비스, INIPAY 오픈웹서비스, 모바일 서비스, ARS 서비스, VA N(Value Added Network) 서비스, 키오스크 및 테이블 오더 서비스를 제공합니다.",
        "Pricing Sections": "KG이니시스는 신용카드, 체크카드, 멤버십 카드 등 다양한 결제수단을 지원하며 현금IC 거래, 간편결제/해외카드 서비스도 제공합니다. 또한, 가맹점 브랜드에 맞는 자체 간편결제 서비스 구축이 가능한 WPAY(더블유페이) 서비스와 최대 60개월까지 월분할 납부가 가능한 이니렌탈 구독 결제도 제공합니다.",
        "Contact Sections": "KG이니시스는 INIPAY PC 및 모바일 결제창 광고 플랫폼과 가맹점 관리자사이트를 통한 B2B 플랫폼 광고를 제공하여 광고 서비스를 제공합니다. 실시간 번역 서비스, 카드 본인확인 서비스 등 부가서비스를 제공하며 일본 결제용 TOtal 솔루션을 통해 한국 및 일본 소비자 대상의 결제 솔루션을 제공합니다.",
        "Footers": "KG이니시스는 시장점유율 1위 전자결제 전문 기업으로 안전한 거래 및 결제 솔루션을 제공하며 다양한 사업영역을 통해 고객에게 종합적인 전자결제 서비스를 제공합니다. KG이니시스는 시장점유율 1위 전자결제 전문 기업으로, 다양한 사업영역을 통해 통합간편결제서비스, INIPAY 오픈웹서비스, 모바일 서비스, ARS 서비스, VA N(Value Added Network) 서비스, 키오스크 및 테이블 오더 서비스를 제공합니다."
    }

# 생성된 랜딩 페이지의 태그를 기반으로 블록 content 생성
curl -X POST http://192.168.0.42:8001/land_section_generate \
-H "Content-Type: application/json" \
-d "{"model" : "AI model 명", "block" : {"섹션 이름" : ["블록들의 태그 리스트"]}, "section_context" : {"섹션" : "맥락유지용 데이터"}}

ex. -d "{   "model" : "EEVE",
    "block" : {
        "Navbars" : ["h1_p_p_p_p", "h2_p_p_p_p", "h3_p_p_p_p"],
        "Hero" : ["h1_h3_p", "h2_h3_li(h3_p)*2", "h1_h2_h3_p"],
        "CTA" : ["h3_li(h3_p)*3", "h3_h1_p_li(h2_p)*3", "h2_h3_p"],
        "Pricing" : ["h3_li(h3_p)*3_p_h3_li(p)*3", "h1_h3_li(p)*3", "h3_p_li(h3_p)*3"],
        "Content" : ["h1_h1_h3_li(h2_h3)*2", "h3_h1_h2_p", "li(h3_p)*3"],
        "Testimonial" : ["h3_li(h3_p)*3_p_h3", "h2_p_h3_li(p)*3_h3", "h1_h3_p_li(h2_h3_p)*2"],
        "FAQ" : ["h1_h3_p", "h2_li(p)*6", "h3_li(p)*5"],
        "Team" : ["h3_h1_h3_p", "h1_h2_p_h3", "h1_p_"],
        "Comparison": ["h3_h2_p_p_h3", "h1_h3_p_h3_p", "p_h3_h1"],
        "Footer" : ["li(p)*7", "h3_li(p)*6", "p_h3_li(p)*5"]
    },
    "section_context" : {
        "Navbars": "KG이니시스는 1998년 설립된 전자결제 선도 기업으로, 시장 점유율 1위를 자랑하며 안전한 결제 서비스를 제공합니다. 다양한 사업 분야를 통해 온라인 및 오프라인 가맹점에 종합적인 솔루션을 제공하며 간편결제, 통합간편결제, VA(Value Added Network) 등 다양한 서비스를 제공합니다.",
        "Hero": "KG이니시스는 업계 최다 기술특허 보유, PCI-DSs 인증 획득, 우수한 재무 안정성을 자랑합니다. 기업 간 전자상거래 플랫폼과 광고플랫폼을 운영하며 디지털 금융 분야에서 입지를 강화하고 있습니다.",
        "CTA": "KG이니시스의 주요 사업으로는 PG(Payment Gateway) 서비스, VA N 서비스, 시너지 서비스가 있으며, 이 중 PG 서비스는 국내 1위 PG사로 안전한 거래 환경을 제공합니다. 다양한 간편결제 솔루션을 통합하여 제공하며 업계 최다 기술특허를 보유하고 있습니다.",
        "Pricing": "KG이니시스의 재무 안정성은 우수하며 경쟁사 대비 월등한 유동성을 보유해 안정적인 고객사 정산 서비스를 제공합니다. INIPAY 모바일 및 오픈웹 결제 서비스, ARS 서비스 등 다양한 전자결제 솔루션을 제공하며 통합간편결제서비스로 가맹점의 결제 과정을 간소화합니다.",
        "Content": "KG이니시스는 광고플랫폼 서비스를 운영하여 B2C 및 B2B 영역에서 다양한 광고 기회를 제공하며 월 350만 명 이상의 결제 고객에게 노출됩니다. 이니렌탈서비스를 통해 렌탈페이+를 도입해 소비자에게 편리한 결제를 제공합니다.",
        "Testimonial": "KG이니시스는 업계 최초로 PCI-DSs 인증 획득하여 글로벌 보안 기준을 준수하며 다양한 간편결제 솔루션을 제공하여 소비자 및 가맹점 모두에게 편리하고 안전한 결제 환경을 제공합니다. 전자결제 선도 기업으로, 시장 점유율 1위를 자랑하며 안전한 결제 서비스를 제공합니다.",
        "FAQ": "KG이니시스는 PG(Payment Gateway) 및 VA(Value Added Network) 서비스 제공, 간편결제 솔루션 통합, 다양한 결제 수단 지원을 통해 원활한 거래를 지원하고 키오스크 및 테이블 오더 서비스를 제공합니다.",
        "Team": "KG이니시스의 주요 사업으로는 PG(Payment Gateway) 서비스, VA N 서비스, 시너지 서비스가 있으며, 이 중 PG 서비스는 국내 1위 PG사로 안전한 거래 환경을 제공합니다. 다양한 간편결제 솔루션을 통합하여 제공하며 업계 최다 기술특허를 보유하고 있습니다.",
        "Comparison": "KG이니시스의 재무 안정성은 우수하며 경쟁사 대비 월등한 유동성을 보유해 안정적인 고객사 정산 서비스를 제공합니다. INIPAY 모바일 및 오픈웹 결제 서비스, ARS 서비스 등 다양한 전자결제 솔루션을 제공하며 통합간편결제서비스로 가맹점의 결제 과정을 간소화합니다.",
        "Footer": "KG이니시스의 재무 안정성은 우수하며 경쟁사 대비 월등한 유동성을 보유해 안정적인 고객사 정산 서비스를 제공합니다. INIPAY 모바일 및 오픈웹 결제 서비스, ARS 서비스 등 다양한 전자결제 솔루션을 제공하며 통합간편결제서비스로 가맹점의 결제 과정을 간소화합니다."}}"

기대 값:
{
    "Navbars": {
        "HTML_Tag": "h1_p_p_p_p",
        "gen_content": "<h1>전자결제 선도 기업 KG이니시스</h1>\r<p>KG이니시스는 1998년 설립된 전자결제 선도 기업으로, 시장 점유율 1위를 자랑하며 안전한 결제 서비스를 제공합니다.</p>\r<p>다양한 사업 분야를 통해 온라인 및 오프라인 가맹점에 종합적인 솔루션을 제공하며 간편결제, 통합간편결제, VA(Value Added Network) 등 다양한 서비스를 제공합니다.</p>"
    },
    "Hero": {
        "HTML_Tag": "h1_h3_p",
        "gen_content": "<h1>KG이니시스, 업계 선도 기술 및 금융 안정성</h1>\r\r<h3>최다 기술특허 보유, PCI-DSS 인증, 강력한 재무 안정성</h3>\r\r<p>KG이니시스는 전자상거래 플랫폼과 광고플랫폼을 운영하며 디지털 금융 분야에서 선도적인 위치를 차지하고 있습니다. 업계에서 가장 많은 기술 특허를 보유하고 있으며, 결제 보안에 대한 국제 표준인 PCI-DSS 인증을 받았습니다. 또한 강력한 재무 안정성을 바탕으로 고객에게 신뢰할 수 있는 서비스를 제공합니다.</p>"
    },
    "CTA": {
        "HTML_Tag": "h3_li(h3_p)*3",
        "gen_content": "<h3>KG이니시스의 주요 사업</h3>\r<ul>\r  <li>\r    <h3>PG(Payment Gateway) 서비스</h3>\r    <p>KG이니시스는 국내 1위 PG사로 안전한 거래 환경을 제공합니다. 다양한 간편결제 솔루션을 통합하여 제공하며 업계 최다 기술특허를 보유하고 있습니다.</p>\r  </li>\r  <li>\r    <h3>VA N 서비스</h3>\r    <!-- 여기에 VA N 서비스에 대한 내용을 추가하세요 -->\r  </li>\r  <li>\r    <h3>시너지 서비스</h3>\r    <!-- 여기에 시너지 서비스에 대한 내용을 추가하세요 -->\r  </li>\r</ul>"
    },
    "Pricing": {
        "HTML_Tag": "h3_li(h3_p)*3_p_h3_li(p)*3",
        "gen_content": "<h3>KG이니시스</h3>\r<ul>\\n  <li>\\n    <h3>재무 안정성 우수</h3>\\n    <p>KG이니시스는 경쟁사 대비 월등한 유동성을 보유하여 안정적인 고객사 정산 서비스를 제공합니다.</p>\\n  </li>\\n  <li>\\n    <h3>다양한 전자결제 솔루션</h3>\\n    <p>INIPAY 모바일 및 오픈웹 결제 서비스, ARS 서비스 등 다양한 전자결제 솔루션을 제공하며 통합간편결제서비스로 가맹점의 결제 과정을 간소화합니다.</p>\\n  </li>\\n</ul>\r\r<h3>KG이니시스의 장점</h3>\r<ul>\\n  <li>\\n    <p>우수한 재무 안정성</p>\\n  </li>\\n  <li>\\n    <p>다양한 전자결제 솔루션 제공</p>\\n  </li>\\n  <li>\\n    <p>통합간편결제서비스로 결제 과정 간소화</p>\\n  </li>\\n</ul>"
    },
    "Content": {
        "HTML_Tag": "h1_h1_h3_li(h2_h3)*2",
        "gen_content": "<h1>콘텐츠</h1>\r\r<h1>이니시스 소개</h1>\r\r<h3>이니시스의 광고 플랫폼 서비스</h3>\r\rKG이니시스는 B2C 및 B2B 영역에서 다양한 광고 기회를 제공하며 월 350만 명 이상의 결제 고객에게 노출되는 광고플랫폼 서비스를 운영합니다.\r\r<h3>이니렌탈서비스의 렌탈페이+</h3>\r\r이니렌탈서비스는 소비자에게 편리한 결제를 제공하기 위해 렌탈페이+를 도입하였습니다.\r\r<ul>\r  <li>\r    <h2>이니시스 광고 플랫폼 서비스</h2>\r    <h3>B2C 및 B2B 영역에서의 다양한 광고 기회</h3>\r  </li>\r  <li>\r    <h2>이니렌탈서비스의 렌탈페이+</h2>\r    <h3>소비자에게 편리한 결제 제공</h3>\r  </li>\r</ul>"
    },
    "Testimonial": {
        "HTML_Tag": "h3_li(h3_p)*3_p_h3",
        "gen_content": "<h3>고객의 말</h3>\r<ul>\r  <li>\r    <h3>KG이니시스는 업계 최초로 PCI-DSS 인증을 획득하여 글로벌 보안 기준을 준수하며 다양한 간편결제 솔루션을 제공하여 소비자 및 가맹점 모두에게 편리하고 안전한 결제 환경을 제공합니다.</h3>\r    <p>전자결제 선도 기업으로, 시장 점유율 1위를 자랑하며 안전한 결제 서비스를 제공합니다.</p>\r  </li>\r  <li>\r    <h3>KG이니시스는 업계 최초로 PCI-DSS 인증을 획득하여 글로벌 보안 기준을 준수하며 다양한 간편결제 솔루션을 제공하여 소비자 및 가맹점 모두에게 편리하고 안전한 결제 환경을 제공합니다.</h3>\r    <p>전자결제 선도 기업으로, 시장 점유율 1위를 자랑하며 안전한 결제 서비스를 제공합니다.</p>\r  </li>\r  <li>\r    <h3>KG이니시스는 업계 최초로 PCI-DSS 인증을 획득하여 글로벌 보안 기준을 준수하며 다양한 간편결제 솔루션을 제공하여 소비자 및 가맹점 모두에게 편리하고 안전한 결제 환경을 제공합니다.</h3>\r    <p>전자결제 선도 기업으로, 시장 점유율 1위를 자랑하며 안전한 결제 서비스를 제공합니다.</p>\r  </li>\r</ul>\r\r<p>KG이니시스는 업계 최초로 PCI-DSS 인증을 획득하여 글로벌 보안 기준을 준수하며 다양한 간편결제 솔루션을 제공하여 소비자 및 가맹점 모두에게 편리하고 안전한 결제 환경을 제공합니다. 전자결제 선도 기업으로, 시장 점유율 1위를 자랑하며 안전한 결제 서비스를 제공합니다.</p>\r\r<h3>KG이니시스의 고객</h3>"
    },
    "FAQ": {
        "HTML_Tag": "h1_h3_p",
        "gen_content": "<h1>FAQ</h1>\r\r                    <h3>KG이니시스에 대해 궁금한 점들</h3>\r\r                    <p>KG이니시스는 PG(Payment Gateway) 및 VA(Value Added Network) 서비스 제공, 간편결제 솔루션 통합, 다양한 결제 수단 지원을 통해 원활한 거래를 지원하고 키오스크 및 테이블 오더 서비스를 제공합니다. \r\r                    </p>"
    },
    "Team": {
        "HTML_Tag": "h3_h1_h3_p",
        "gen_content": "<h3>팀</h3>\r\r                    <h1>KG이니시스 주요 사업</h1>\r\r                    <h3>PG(결제 게이트웨이) 서비스</h3>\r                    <p>KG이니시스의 주요 사업으로는 PG(Payment Gateway) 서비스, VA N 서비스, 시너지 서비스가 있으며, 이 중 PG 서비스는 국내 1위 PG사로 안전한 거래 환경을 제공합니다. 다양한 간편결제 솔루션을 통합하여 제공하며 업계 최다 기술특허를 보유하고 있습니다.</p>\r\r                    <h3>VA N 서비스</h3>\r                    <p>KG이니시스의 주요 사업으로는 VA N 서비스가 있으며, 이는 가상계좌 서비스로 기업과 개인에게 편리한 결제 수단을 제공합니다. 다양한 간편결제 솔루션을 통합하여 제공하며 업계 최다 기술특허를 보유하고 있습니다.</p>\r\r                    <h3>시너지 서비스</h3>\r                    <p>KG이니시스의 주요 사업으로는 시너지 서비스가 있으며, 이는 기업과 개인에게 다양한 결제 솔루션을 제공합니다. 다양한 간편결제 솔루션을 통합하여 제공하며 업계 최다 기술특허를 보유하고 있습니다.</p>"
    },
    "Comparison": {
        "HTML_Tag": "h3_h2_p_p_h3",
        "gen_content": "<h3>KG이니시스</h3>\r<h2>재무 안정성 및 유동성</h2>\r<p>KG이니시스는 우수한 재무 안정성을 자랑하며 경쟁사 대비 뛰어난 유동성을 보유하고 있습니다. 이를 통해 안정적인 고객사 정산 서비스를 제공합니다.</p>\r\r<p>INIPAY 모바일 및 오픈웹 결제 서비스, ARS 서비스 등 다양한 전자결제 솔루션을 제공하며 통합간편결제서비스로 가맹점의 결제 과정을 간소화합니다.</p>\r\r<h3>KG이니시스와 INIPAY 비교</h3>"
    },
    "Footer": {
        "HTML_Tag": "li(p)*7",
        "gen_content": "<ul>\r  <li>    <p>KG이니시스는 재무 안정성이 우수하며 경쟁사 대비 월등한 유동성을 보유하고 있어 안정적인 고객사 정산 서비스를 제공합니다.</p></li>\r  <li>    <p>INIPAY 모바일 및 오픈웹 결제 서비스, ARS 서비스 등 다양한 전자결제 솔루션을 제공합니다.</p></li>\r  <li>    <p>통합간편결제서비스로 가맹점의 결제 과정을 간소화합니다.</p></li>\r</ul>"
    }
}

land_section_generate의 주요 역할
1. PDF를 읽어서 str 변환 (PDF2TEXT / param : pdf_list, PDF cdn 경로 리스트)
2. PDF data를 요약 summary (OllamaSummaryClient.store_chunks / Param : data str, PDF 스트링 데이터 / Param : model_max_token : 8192 int 모델의 맥스토큰, final_summary_length : 6000 int 요약본 길이, max_tokens_per_chunk : 6000 int, 얼마나 요약할지 지정)
3. summary한 데이터를 토대로 섹션 구조 지정(OllamaMenuClient.section_structure_create_logic / param : summary str 요약데이터 입력)
4. 뽑아낸 섹션 구조를 각 섹션 단위로 data 생성. (OllamaBlockRecommend.generate_block_content / param : summary str, 요약데이터 / param : block_list : block dict 섹션별 블록들 태그 리스트까지.)



