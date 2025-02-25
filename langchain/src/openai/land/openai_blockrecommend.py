import asyncio
from typing import List, Dict, Any
import re
import difflib
import json

class OpenAIBlockSelector:
    def __init__(self, batch_handler):
        self.batch_handler = batch_handler

    async def send_request(self, prompt: str, max_tokens: int = 50) -> str:
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.1,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=300  # 적절한 타임아웃 값 설정
        )
        return response

    async def select_block(self,
        section_context: str,
        block_list: Dict[str, str],
        max_tokens: int = 50) -> Dict[str, Any]:
        
        only_section_context = section_context[1]
        tag_slice = list(block_list[1].values())

        for attempt in range(3):  # 최대 3번 시도
            try:
                prompt = f"""
                [System]
                You are an AI assistant that selects appropriate HTML tags for website sections. Follow these instructions precisely:

                1. Read the given section context and tag list.
                2. Select ONE tag from the list that best represents the section context.
                3. Return ONLY a JSON object with the key "selected_tag" and the chosen tag as its value.
                4. Do NOT include any additional text or explanations.
                5. Ensure the output is a valid JSON object.
                6. Do NOT copy or use the example outputs directly. Generate a new, appropriate response based on the given input.

                --- EXAMPLE INPUTS AND OUTPUTS (FOR REFERENCE ONLY) ---
                Example Input 1:
                section context = "회사 소개: 우리는 혁신적인 기술 솔루션을 제공하는 선도적인 기업입니다."
                tag list = ["h1_p", "h2_li_p", "h3_h5_p"]

                Example Output 1:
                {{"selected_tag": "h1_p"}}

                Example Input 2:
                section context = "제품 목록: 1. 스마트폰 2. 태블릿 3. 노트북"
                tag list = ["p_li", "h2_ul_li", "h3_ol_li"]

                Example Output 2:
                {{"selected_tag": "h2_ul_li"}}
                --- END OF EXAMPLES ---

                [User]
                section context = {only_section_context}
                tag list = {tag_slice}

                [Assistant]
                # YOUR RESPONSE STARTS HERE (ONLY INCLUDE THE JSON OBJECT):
                """

                result = await self.send_request(prompt, max_tokens)
                
                selected_Html_tag_json = self.extract_json(result.data.generations[0][0].text)
                print(f"생성 직후 tag : {selected_Html_tag_json}")
                if selected_Html_tag_json and 'selected_tag' in selected_Html_tag_json:
                    selected_Html_tag_str = selected_Html_tag_json['selected_tag']
                    
                    if selected_Html_tag_str in tag_slice:
                        section_name = section_context[0]
                        reversed_block_dict = {value: key for key, value in block_list[1].items()}
                        
                        return {
                            'Section_name': section_name,
                            'Block_id': reversed_block_dict[selected_Html_tag_str],
                            'HTML_Tag': selected_Html_tag_str
                        }
                
                print(f"Attempt {attempt + 1} failed. Retrying...")
            
            except Exception as e:
                print("block recommend error :", e)
                if (attempt + 1) == 3:
                    return {
                        'Section_name': section_context[0],
                        'Block_id': list(block_list[1].keys())[0],
                        'HTML_Tag': tag_slice[0]
                    }
                    
            # 3번 시도 후에도 실패하면 기본값 반환
            print("All attempts failed. Returning default response.")
        return {
            'Section_name': section_context[0],
            'Block_id': list(block_list[1].keys())[0],
            'HTML_Tag': tag_slice[0]
        }

    async def select_block_batch(
        self,
        section_names_n_contents: List[str],
        section_names_n_block_lists: List[Dict[str, str]],
        max_tokens: int = 50) -> List[Dict[str, Any]]:
    
        select_block_results = []
        for section_name, block_list in zip(section_names_n_contents, section_names_n_block_lists):
            
            
            # NOTE 250219 : 데이터가 들어오는 만큼 뭉텅이로 보낼 수 있게 설계
            temp_section_list = list(section_name.items())
            temp_block_list = list(block_list.items())
            
            for n_temp_list, n_temp_block_list in zip(temp_section_list, temp_block_list):
                
                select_block_result = await self.select_block(n_temp_list, n_temp_block_list, max_tokens)
                select_block_results.append(select_block_result)
        
        # NOTE 250220: batch 방식은 추후 적용
        # tasks = []
        # tasks = [
        #     self.select_block(section_name, block_list)
        #     for section_name, block_list in zip(section_names_n_contents, section_names_n_block_lists)
        # ]
        # task = self.select_block(n_temp_list, n_temp_block_list)
        #         tasks.append(task)
        # return await asyncio.gather(*tasks)
        
        return select_block_results
            


    def extract_emmet_tag(self, text: str) -> str:
        raw_value = text.strip("**").strip().replace("**", "")
        raw_value = raw_value.split("\n", 1)[0]  # 첫 번째 줄만 사용
        allowed_chars = set("hlip123456789_()*+")
        filtered = "".join(ch for ch in raw_value if ch in allowed_chars)
        return filtered if filtered else text


    # NOTE 250219 : 여기가 안 돌아간다는 것. keyError
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

    def extract_text(self, result):
        if result.success and result.data.generations:
            return result.data
        else:
            return "텍스트 생성 실패"
        
    def extract_json(self, text):
    # 가장 바깥쪽의 중괄호 쌍을 찾습니다.
        print(f"block recommend text : {text}")
        # re.sub(r'')
        text = re.sub(r'[\n\r\\\\/]', '', text, flags=re.DOTALL)
        json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            # Handle case where only opening brace is found
            json_str = re.search(r'\{.*', text, re.DOTALL)
            if json_str:
                json_str = json_str.group() + '}'
            else:
                return None

        # Balance braces if necessary
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try a more lenient parsing approach
            try:
                return json.loads(json_str.replace("'", '"'))
            except json.JSONDecodeError:
                return None
