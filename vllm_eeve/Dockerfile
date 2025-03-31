ARG CUDA_VERSION=12.6.0
ARG PYTHON_VERSION=3.10
# GPU 사용을 위한 CUDA 베이스 이미지 선택 (CUDA 버전은 필요에 따라 조정)
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base


# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-pip \
    git \
    ca-certificates \
# NOTE : 하위 3가지는 extra_body={"guided_json" ~ }를 위한 패키지지
    curl \
    python3-dev \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app
ENV VLLM_GPU_MEMORY_UTILIZATION=1.0
# requirements.txt 복사 (여기에는 vllm, torch, aiohttp, fastapi 등 필요한 라이브러리들이 포함되어 있어야 합니다.)
COPY requirements.txt .
COPY /models/EEVE-Korean-Instruct-10.8B-v1.0 /usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0


RUN apt-get update && apt-get install -y python3-pip

# pip 업그레이드 후 requirements.txt 설치
RUN pip install --upgrade pip && pip3 install --only-binary=:all: -r requirements.txt

# 애플리케이션 코드 복사 (필요한 경우)
COPY . .

# API 서버가 사용할 포트 노출 (예: 8000)
EXPOSE 8002

# vLLM 추론 서버 실행 (추가 인자가 필요하면 환경변수나 CMD 인자에 포함)

# CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--port=8002", "--host=0.0.0.0"]

# NOTE_250311 : run_server.py 사용을 위한 테스트
#               api_server는 CLI로 실행하는게 낫다는 판단
#               제대로 열어줄려면 docker-compose.yml처럼 --model도 같이 넣어줘야 함. 아니면 안 열림
#               이렇게 모델 지정해서 열기 때문에 포트를 바꾸지 않으면 하나의 모델만 사용 가능
# CMD ["python3", "/app/models/src/run_server.py"]