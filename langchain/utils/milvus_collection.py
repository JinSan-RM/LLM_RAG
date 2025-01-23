import json, os, re
import pandas as pd
# from openai import OpenAI
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from fastapi import FastAPI
from datetime import datetime

import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# OPENAI_KEY = os.environ['OPENAI_API_KEY']
class MilvusDataHandler:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_KEY)

    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def extract_section_type(self, tags):
        # Shaw_250121_여기서 "Multi_step" 뺄께요!!
        predefined_sections = [
            "Hero_Header", "Feature", "CTA", "Contact", "Pricing", "Stats", "Content", 
            "Testimonial", "FAQ", "Logo", "Team", "Gallery",
            "Timeline", "Comparison", "Countdown"
        ]
        
        for section in predefined_sections:
            if section in tags:  # 태그에서 소문자로 검색
                return section
        return "Unknown"  # 기본값

    def extract_emmet_tags(self, tags):
        """
        Emmet 태그를 추출하고 원본 구조를 포함하여 반환.
        """
        # Emmet 태그 정의
        emmet_elements = ["li", "h1", "h2", "h3", "h4", "h5", "p"]

        # 전체 태그 구조 유지
        full_structure = tags.strip()

        # Emmet 태그 추출
        extracted_tags = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", tags)
        filtered_tags = [tag for tag in extracted_tags if tag in emmet_elements]

        # 중복 제거
        unique_tags = sorted(set(filtered_tags))

        # 결과를 딕셔너리로 반환
        extracted_tags = ", ".join(extracted_tags)  # 리스트를 문자열로 변환
        return full_structure, extracted_tags 
        

    def extract_additional_tags(self, tags):
        additional_patterns = ["sp", "bt", "img_"]
        extracted_tags = []
        for tag in tags.split(", "):
            if any(pattern in tag for pattern in additional_patterns):
                extracted_tags.append(tag)
        return ", ".join(extracted_tags)  # 리스트를 문자열로 변환

    def process_data(self, json_data):
        processed_data = []

        for item in json_data:
            template_id = item.get("template_id")
            concatenated_tags = item.get("concatenated_tags", "")

            # 섹션 유형 및 추가 태그
            section_type = self.extract_section_type(concatenated_tags)
            additional_tags = self.extract_additional_tags(concatenated_tags)

            # Emmet 태그 추출
            emmet_tag, include_tag = self.extract_emmet_tags(concatenated_tags)

            # 태그 임베딩 생성
            embedding = [self.text_embedding(concatenated_tags)]
        

            print(embedding)
            
            # 데이터 구성
            processed_data.append([
                template_id,
                section_type,
                emmet_tag,
                include_tag,
                additional_tags,
                embedding,
                0,
                0,  # 초기 popularity 값
                ""  # layout_type은 현재 비어 있음

            ])
            # print("\n", processed_data)

        return processed_data

    def text_embedding(self, question: str):
        print(f"질문 임베딩 생성: {question}")
        response = self.client.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        )
        # print(f"생성된 임베딩: {response.data[0].embedding}")
        return response.data[0].embedding
    
    def create_milvus_collection(self):
        connections.disconnect(alias='default')
        # milvus DB connection 점검
        
        connections.connect(alias="default", host="172.19.0.6", port="19530")
        if utility.has_collection('block_collection'):
            collection = Collection('block_collection')
            collection.drop()
            print("이미 있지만 다시 지웠다가 생성.")

        fields = [
            FieldSchema(name="template_id", dtype=DataType.VARCHAR, max_length=50, is_primary=True),
            FieldSchema(name="section_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="emmet_tag", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="include_tag", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="additional_tags", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="AI_popularity", dtype=DataType.INT64),
            FieldSchema(name="User_popularity", dtype=DataType.INT64),
            FieldSchema(name="layout_type", dtype=DataType.VARCHAR, max_length=50)
        ]

        schema = CollectionSchema(fields, description="Block data for recommendations")
        collection = Collection(name="block_collection", schema=schema)
        
        self.create_index(collection)

        collection.load()
        
        return collection
    def create_index(self, collection):
        index_params = {
            "index_type": "IVF_FLAT",  # 인덱스 유형: IVF_FLAT, IVF_SQ8, HNSW 등
            "metric_type": "L2",  # 거리 계산 방식: L2 (유클리디안 거리), IP (내적)
            "params": {"nlist": 128}  # 적절한 파라미터 설정
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("Index created successfully!")

    def insert_data_to_milvus(self, collection, data):
        for item in data:
            template_id, section_type, emmet_tag, include_tag, additional_tags, embedding, AIpopularity, popularity, layout_type = item
            # print("\n", template_id, section_type, emmet_tag, additional_tags, embedding, popularity, layout_type, created_at , " \ninsert data\n")
            collection.insert([
                [template_id], [section_type], [emmet_tag], [include_tag], [additional_tags], embedding, [AIpopularity], [popularity], [layout_type]
            ])
        collection.flush()


