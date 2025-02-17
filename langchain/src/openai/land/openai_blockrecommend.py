import asyncio
from typing import List, Dict, Any
import re
import difflib

class OpenAIBlockSelector:
    def __init__(self, batch_handler):
        self.batch_handler = batch_handler

    async def send_request(self, prompt: str) -> str:
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.1,
                "top_p": 1.0,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=60  # 적절한 타임아웃 값 설정
        )
        return response

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
        당신은 HTML 태그 선택을 돕는 AI 어시스턴트입니다. 다음 지침을 따르세요:
        1. 섹션에 어울리는 태그를 태그 리스트 중에서 선택하세요.
        2. 단 하나의 태그만 반환하세요.
        3. 태그 HTML 구조는 그대로 유지하세요.
        4. 다른 텍스트는 출력하지 마세요.
        5. 반드시 태그 리스트 안의 데이터 중 하나만 골라 출력하세요.

        User: 다음 태그 리스트에서 적절한 태그를 선택해주세요:
        {block_list}
        """
        
        
        raw_json = await self.send_request(prompt)
        b_id = self.find_key_by_value(block_list, raw_json)
        if b_id is None:
            raise ValueError(f"매칭되는 b_id가 없습니다: ")
        
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
