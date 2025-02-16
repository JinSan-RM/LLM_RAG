import asyncio
from typing import List, Dict, Any
import re
import difflib

class BlockSelector:
    def __init__(self, batch_handler):
        self.batch_handler = batch_handler

    async def send_request(self, prompt: str) -> str:
        request = {
            "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
            "prompt": prompt,
            "temperature": 0.1,
        }
        response = await self.batch_handler.process_single_request(request, request_id=0)
        if response.success:
            return response.data.generations.text.strip()
        else:
            raise RuntimeError(f"API 요청 실패: {response.error}")

    async def select_block(self, section_context: str, block_list: Dict[str, str]) -> Dict[str, Any]:
        tag_slice = list(block_list.values())
        """prompt = f
        System: 
        당신은 HTML 태그 선택을 돕는 AI 어시스턴트입니다. 다음 지침을 따르세요:
        1. {section_name} 섹션에 어울리는 태그를 태그 리스트 중에서 선택하세요.
        2. 단 하나의 태그만 반환하세요.
        3. 태그 HTML 구조는 그대로 유지하세요.
        4. 다른 텍스트는 출력하지 마세요.
        5. 반드시 태그 리스트 안의 데이터 중 하나만 골라 출력하세요.

        User: 다음 태그 리스트에서 적절한 태그를 선택해주세요:
        {tag_slice}
        """
        # 원래 section_name 인데 이걸 section_context로로 변경
        # 왜냐면 이미 1차적으로 섹션이 정해진 것이기 떄문에, 컨텐츠를 잘 표현할 리스트를 찾는게 맞음
        prompt = f"""
        System: 
        You are an AI assistant that helps you select HTML tags similar to the emmet format. 
        
        Instructions:
        
        1. Select a tag that best represents section_content from the tag_list in User input.
        2. Select only one tag.
        3. Return the selected tag as is.
        4. Do not print any text other than the selected tag.
        5. Check whether you selected one of the data in the tag list, or select it again and return it.
        
        User input: 
        section_content : {section_context}
        tag_list : {tag_slice}
        
        """        
        
        
        raw_json = await self.send_request(prompt)
        b_id = self.find_key_by_value(block_list, raw_json)
        if b_id is None:
            raise ValueError(f"매칭되는 b_id가 없습니다: {section_name}")
        
        b_value = self.extract_emmet_tag(block_list[b_id])
        
        return {
            'HTML_Tag': b_value,
            'Block_id': b_id
        }

    async def select_block_batch(self, section_names: List[str], block_lists: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        tasks = [
            self.select_block(section_name, block_list)
            for section_name, block_list in zip(section_names, block_lists)
        ]
        return await asyncio.gather(*tasks)

    def extract_emmet_tag(self, text: str) -> str:
        raw_value = text.strip("**").strip().replace("**", "")
        raw_value = raw_value.split("\n", 1)[0]  # 첫 번째 줄만 사용
        allowed_chars = set("hlip123456789_()*+")
        filtered = "".join(ch for ch in raw_value if ch in allowed_chars)
        return filtered if filtered else text

    def find_key_by_value(self, mapping: dict, target_value: str):
        for key, value in mapping.items():
            if value == target_value:
                return key

        # 유사도가 가장 높은 키를 찾습니다.
        closest_match = None
        closest_score = -float("inf")
        for key, value in mapping.items():
            score = difflib.SequenceMatcher(None, value, target_value).ratio()
            if score > closest_score:
                closest_match = key
                closest_score = score
        print(f"closest_match: {closest_match}, closest_score: {closest_score}")
        return closest_match if closest_match else None

    def process_header_footer(self, data_list: Dict[str, str], ctx_value: str) -> Dict[str, Any]:
        first_item = next(iter(data_list.items()))
        b_id, b_value = first_item
        b_value = self.extract_emmet_tag(b_value)
        return {
            "HTML_Tag": b_value,
            "Block_id": b_id,
            "gen_content": ctx_value,
        }
