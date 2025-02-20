'''
설정 파일 (모델 및 API 설정)
'''
# config.py
import torch
import os
from pymilvus import connections

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OLLAMA_IP = os.environ.get("ollama_IP")
# print(f"OLLAMA_IP: {OLLAMA_IP}")

# config.py
OLLAMA_API_URL = "http://ollama:11434/"

# connections.connect(host=OLLAMA_IP, port='19530')
# LangChain 관련 설정

MAX_LENGTH = 512

MILVUS_HOST = 'milvus-standalone'
MILVUS_PORT = '19530'

'''
ETCD_ENDPOINTS = "etcd:2379"
MINIO_ADDRESS = "minio:9000"
'''


