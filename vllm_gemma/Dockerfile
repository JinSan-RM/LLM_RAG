# vLLM 공식 이미지 사용 (최신 버전 기준)
FROM vllm/vllm-openai:latest

# 작업 디렉토리 설정
WORKDIR /app

# 모델 파일을 컨테이너에 복사 (선택 사항)
# 로컬에서 모델을 미리 다운로드한 경우에만 사용
# COPY /models/gemma-3-1b-it /usr/local/bin/models/gemma-3-1b-it
COPY /models/gemma-3-4b-it /usr/local/bin/models/gemma-3-4b-it
COPY /models/gemma-3-12b-it /usr/local/bin/models/gemma-3-12b-it

# 추가적인 Python 패키지 설치 (필요 시)
# 예: RUN pip install some-package

# 환경 변수 설정 (선택 사항)
ENV NVIDIA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# vLLM 서버 실행 명령어 (docker-compose에서 오버라이드 가능)
# CMD ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "/usr/local/bin/models/gemma-3-4b-it", "--port", "8022", "--dtype", "float16"]

# ========== 250403 이전 버전 ==============

# ARG CUDA_VERSION=12.6.0
# ARG PYTHON_VERSION=3.10
# # GPU 사용을 위한 CUDA 베이스 이미지 선택 (CUDA 버전은 필요에 따라 조정)
# FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base

# # 필수 시스템 패키지 설치
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     cmake \
#     python3-pip \
#     git \
#     ca-certificates \
# # NOTE : 하위 3가지는 extra_body={"guided_json" ~ }를 위한 패키지지
#     curl \
#     python3-dev \
#     gcc \
#  && rm -rf /var/lib/apt/lists/*

# # 작업 디렉토리 설정
# WORKDIR /app
# ENV VLLM_GPU_MEMORY_UTILIZATION=1.0
# # requirements.txt 복사 (여기에는 vllm, torch, aiohttp, fastapi 등 필요한 라이브러리들이 포함되어 있어야 합니다.)
# COPY requirements.txt .

# # NOTE 250319 : gemma-2-2b-it 테스트
# # COPY /models/gemma-2-2b-it /usr/local/bin/models/gemma-2-2b-it

# # NOTE 250320 : gemma-2-9b-it / EXAONE-Deep-2.4B / EXAONE-Deep-7.8B 테스트
# # COPY /models/gemma-2-9b-it /usr/local/bin/models/gemma-2-9b-it
# # COPY /models/EXAONE-Deep-2.4B /usr/local/bin/models/EXAONE-Deep-2.4B
# # COPY /models/EXAONE-Deep-7.8B /usr/local/bin/models/EXAONE-Deep-7.8B

# # NOTE 250321
# COPY /models/gemma-3-4b-it /usr/local/bin/models/gemma-3-4b-it
# # COPY /models/gemma-3-12b-it /usr/local/bin/models/gemma-3-12b-it

# # NOTE 250403
# # COPY /models/gemma-3-1b-it /usr/local/bin/models/gemma-3-1b-it

# RUN apt-get update && apt-get install -y python3-pip

# # pip 업그레이드 후 requirements.txt 설치
# RUN pip install --upgrade pip && pip3 install --only-binary=:all: -r requirements.txt

# # 애플리케이션 코드 복사 (필요한 경우)
# COPY . .

# # API 서버가 사용할 포트 노출 (예: 8000)
# EXPOSE 8022

# # vLLM 추론 서버 실행 (추가 인자가 필요하면 환경변수나 CMD 인자에 포함)

# CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--port=8022", "--host=0.0.0.0"]

# # NOTE_250311 : run_server.py 사용을 위한 테스트
# #               api_server는 CLI로 실행하는게 낫다는 판단
# #               제대로 열어줄려면 docker-compose.yml처럼 --model도 같이 넣어줘야 함. 아니면 안 열림
# #               이렇게 모델 지정해서 열기 때문에 포트를 바꾸지 않으면 하나의 모델만 사용 가능
# # CMD ["python3", "/app/models/src/run_server.py"]