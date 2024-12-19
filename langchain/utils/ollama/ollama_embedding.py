import requests
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from config.config import OLLAMA_API_URL
import numpy as np
from langchain.embeddings.base import Embeddings

MAX_LENGTH = 512  # 모델에서 허용하는 최대 입력 길이 설정

class TextData(BaseModel):
    question: str
    answer: str
    metadata: dict

# def get_embedding_from_ollama(text: str) -> List[float]:
#     headers = {
#         "Content-Type": "application/json"
#     }

#     data = {
#         "model": "nomic-embed-text",  # 사용하려는 모델 이름
#         "input": text  # 임베딩할 텍스트
#     }
#     print(f"Input text: {text}")

#     # Ollama API에 POST 요청
#     try:
#         response = requests.post(OLLAMA_API_URL + 'api/embeddings', headers=headers, json=data)
#         print(response, "<=====response")

#         if response.status_code == 200:
#             response_data = response.json()
#             embedding = response_data.get("embedding", [])
#             print("Generated Embedding:", embedding)
#             return embedding  # 성공 시 임베딩 반환
#         else:
#             # 실패 시 로그와 기본값 반환
#             print("Error:", response.json())
#             return []  # 빈 리스트 반환
#     except requests.exceptions.RequestException as e:
#         # 요청 예외 처리
#         print(f"Request failed: {e}")
#         return []  # 요청 실패 시 빈 리스트 반환


# import sentencepiece as spm

# def split_text_by_tokens(text: str, max_tokens: int, tokenizer_model: str = "tokenizer.model") -> List[str]:
#     """
#     SentencePiece 토크나이저를 사용해 텍스트를 토큰 단위로 분할.
#     """
#     sp = spm.SentencePieceProcessor()
#     sp.load(tokenizer_model)  # tokenizer.model 파일 로드

#     tokens = sp.encode(text, out_type=str)  # 텍스트를 토큰 리스트로 변환
#     chunks = []
#     current_chunk = []

#     for token in tokens:
#         current_chunk.append(token)
#         if len(current_chunk) >= max_tokens:
#             chunks.append("".join(current_chunk))
#             current_chunk = []

#     # 마지막 청크 처리
#     if current_chunk:
#         chunks.append("".join(current_chunk))

#     return chunks

# def get_embedding_from_ollama(text: str, max_tokens: int = 512) -> List[List[float]]:
#     """
#     Ollama API를 호출해 긴 텍스트를 분할한 후 임베딩 생성.
#     """
#     headers = {
#         "Content-Type": "application/json"
#     }

#     # 텍스트 분할
#     text_chunks = split_text_by_tokens(text, max_tokens)
#     embeddings = []

#     # 각 텍스트 조각에 대해 API 호출
#     for idx, chunk in enumerate(text_chunks):
#         data = {
#             "model": "nomic-embed-text",  # 사용 모델
#             "input": chunk  # 텍스트 조각
#         }
#         print(f"Processing chunk {idx + 1}/{len(text_chunks)}: {chunk}")

#         try:
#             response = requests.post(OLLAMA_API_URL + 'api/embed', headers=headers, json=data)
#             if response.status_code == 200:
#                 response_data = response.json()
#                 embedding = response_data.get("embedding", [])
#                 embeddings.append(embedding)
#             else:
#                 print("Error:", response.json())
#         except requests.exceptions.RequestException as e:
#             print(f"Request failed for chunk {idx + 1}: {e}")
    
#     # 모든 조각의 임베딩 반환
#     return embeddings

def get_embedding_from_ollama(text: str) -> List[float]: 
    headers = {"Content-Type": "application/json"} 
    if isinstance(text, list): 
        text = " ".join(map(str, text)) # 리스트의 각 요소를 문자열로 변환 후 합침 
    data = {"model": "nomic-embed-text", "prompt": text} 
    try: 
        response = requests.post(OLLAMA_API_URL + 'api/embeddings', headers=headers, json=data)
        response.raise_for_status() # 상태 코드가 4xx/5xx일 경우 예외 발생 
    except requests.RequestException as e: 
        print(f"[ERROR] Failed to get embedding from Ollama API: {str(e)}")
        raise 
    if response.status_code == 200: 
        embedding = response.json().get("embedding", []) 
        if not embedding: 
            print("[ERROR] Empty embedding received.") 
            raise ValueError("Failed to generate embedding: embedding is empty.") 
        print("[INFO] Embedding success") 
        return embedding 
    else: 
        error_message = response.json().get("error", "Unknown error") 
        print(f"[ERROR] Ollama API responded with error: {error_message}") 
        raise ValueError(f"Failed to get embedding from Ollama: {error_message}")


def get_max_tokens(text: str, model: str) -> int:
    """
    Ollama 서버로 토큰 수를 확인하는 요청을 보냄
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": text
    }
    try:
        response = requests.post(f"{OLLAMA_API_URL}/tokens", headers=headers, json=payload)
        response.raise_for_status()
        token_count = response.json().get("tokens", 0)
        return token_count
    except requests.exceptions.RequestException as e:
        print(f"Error getting token count: {e}")
        return 0

def split_text_by_server(text: str, max_tokens: int) -> list:
    """
    텍스트를 Ollama 서버의 최대 토큰 길이 기준으로 분할
    """
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        test_chunk = f"{current_chunk} {word}".strip()
        token_count = get_max_tokens(test_chunk, model="nomic-embed-text")
        
        if token_count > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = word
        else:
            current_chunk = test_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def embedding_from_ollama(text: str, max_tokens: int = 512) -> list:
    """
    Ollama 서버에 텍스트를 분할하여 요청하고 임베딩 생성
    """
    headers = {"Content-Type": "application/json"}
    text_chunks = split_text_by_server(text, max_tokens)
    embeddings = []

    for idx, chunk in enumerate(text_chunks):
        payload = {
            "model": "nomic-embed-text",
            "input": chunk
        }
        print(f"Processing chunk {idx + 1}/{len(text_chunks)}: {chunk}")

        try:
            response = requests.post(f"{OLLAMA_API_URL}/embed", headers=headers, json=payload)
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            embeddings.append(embedding)
        except requests.exceptions.RequestException as e:
            print(f"Request failed for chunk {idx + 1}: {e}")
    
    return embeddings
class OllamaEmbeddings(Embeddings):
    def embed_query(self, text: str) -> List[float]:
        if isinstance(text, dict) and "query" in text:
            text = text["query"] # dict에서 query 키의 값을 추출 
        print(f"embed_query start {text} and type : {type(text)}") 
        return get_embedding_from_ollama(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [get_embedding_from_ollama(text) for text in texts]
