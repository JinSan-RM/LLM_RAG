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
