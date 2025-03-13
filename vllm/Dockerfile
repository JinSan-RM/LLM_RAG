ARG CUDA_VERSION=12.6.0
ARG PYTHON_VERSION=3.10
# GPU 사용을 위한 CUDA 베이스 이미지 선택 (CUDA 버전은 필요에 따라 조정)
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base


# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    # python3.11-dev \
    python3-pip \
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app
ENV VLLM_GPU_MEMORY_UTILIZATION=1.0
# requirements.txt 복사 (여기에는 vllm, torch, aiohttp, fastapi 등 필요한 라이브러리들이 포함되어 있어야 합니다.)
COPY requirements.txt .
COPY /models/EEVE-Korean-Instruct-10.8B-v1.0 /usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0
# pip 업그레이드 후 requirements.txt 설치
RUN pip3 install --upgrade pip && pip3 install --only-binary=:all: -r requirements.txt

# 애플리케이션 코드 복사 (필요한 경우)
COPY . .

# API 서버가 사용할 포트 노출 (예: 8000)
EXPOSE 8002

# vLLM 추론 서버 실행 (추가 인자가 필요하면 환경변수나 CMD 인자에 포함)
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--port=8002", "--host=0.0.0.0"]
