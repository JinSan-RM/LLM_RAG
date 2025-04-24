import asyncio
from typing import List, Dict, Any
import re
import difflib
import json
import random

class OpenAIBlockSelector:
    def __init__(self, batch_handler):
        self.batch_handler = batch_handler

    async def send_request(self, sys_prompt: str, usr_prompt: str, max_tokens: int = 50, extra_body: dict = None) -> str:
        model = "/usr/local/bin/models/gemma-3-4b-it"
        try:
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                    # "prompt": prompt,\
                    "model": model,
                    "sys_prompt": sys_prompt,
                    "usr_prompt": usr_prompt,
                    "extra_body": extra_body,
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
        except asyncio.TimeoutError:
            print("[ERROR] Request timed out")
            return "Error: Request timed out after 300 seconds"
        except Exception as e:
            print(f"[ERROR] Unexpected error: {str(e)}")
            return f"Error: {str(e)}"

    async def select_block(self,
        section_context: Dict[str, str],
        block_list: Dict[str, str],
        max_tokens: int = 50) -> Dict[str, Any]:

        only_section_context = section_context[1]
        tag_slice = list(block_list[1].values())

        
        for attempt in range(3):  # 최대 3번 시도
            try:
                sys_prompt = f"""
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
                """
                usr_prompt = f"""
                section context = {only_section_context}
                tag list = {tag_slice}
                """
                extra_body = {
                    "guided_json": {
                        "type": "object",
                        "properties": {
                            "selected_tags": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": tag_slice    
                                },
                                "minItems": 2,
                                "maxItems": 2
                            }
                        },
                        "required": ["selected_tag"]
                    }
                }
                
                result = await self.send_request(
                    sys_prompt=sys_prompt, 
                    usr_prompt=usr_prompt, 
                    max_tokens=max_tokens, 
                    extra_body=extra_body
                    )
                
                selected_Html_tag_json = self.extract_json(result.data['generations'][0][0]['text'])

                if selected_Html_tag_json and 'selected_tags' in selected_Html_tag_json:
                    selected_Html_tag_list = selected_Html_tag_json['selected_tags']

                    
                    LLM_selected_results = []
                    for selected_Html_tag_str in selected_Html_tag_list:
                        if selected_Html_tag_str in tag_slice:
                            section_name = section_context[0]
                            reversed_block_dict = {value: key for key, value in block_list[1].items()}
                            
                            LLM_selected_dict = {
                                'Section_name': section_name,
                                'Block_id': reversed_block_dict[selected_Html_tag_str],
                                'HTML_Tag': selected_Html_tag_str
                            }
                            
                            LLM_selected_results.append(LLM_selected_dict)
                            
                    return LLM_selected_results

            except Exception as e:
                
                print(f"[DEBUG] LLM.select() was not working. Maybe the block recommendation is just 1.{e}")

                random_index = random.randint(0, len(block_list[1].keys()) - 1)

                block_keys = list(block_list[1].keys())
                selected_block_id = block_keys[random_index]
                selected_html_tag = tag_slice[random_index] if len(tag_slice) > random_index else tag_slice[0]  # tag_slice 길이 확인

                return [{
                    'Section_name': section_name,
                    'Block_id': selected_block_id,
                    'HTML_Tag': selected_html_tag
                }]

    async def select_block_batch(
        self,
        section_names_n_contents: List[str],
        section_names_n_block_lists: List[Dict[str, str]],
        max_tokens: int = 50) -> List[Dict[str, Any]]:


        # NOTE 250424 : 우선 한 번 더 감싸는거 배제함. 추후 batch에서 문제 생기면 복원 예정
        select_block_results = []
        for section_name, block_list in zip(section_names_n_contents, section_names_n_block_lists):

            # print("section_name : ", section_name)
            # print("block_list : ", block_list)

            # NOTE 250219 : 데이터가 들어오는 만큼 뭉텅이로 보낼 수 있게 설계
            temp_section_list = list(section_name.items())
            temp_block_list = list(block_list.items())

            for n_temp_list, n_temp_block_list in zip(temp_section_list, temp_block_list):
                
                select_block_result = await self.select_block(n_temp_list, n_temp_block_list, max_tokens)
                select_block_results.append(select_block_result)

        # NOTE 250220: batch 방식은 추후 적용
        return select_block_results
            
    async def select_block_randomly(self, block_list: Dict[str, str]) -> Dict[str, Any]:

        section_name = block_list[0]
        block_pool = list(block_list[1].items())
        tag_slice = list(block_list[1].values())

        try:

            randomly_selected_id_n_tags = random.sample(block_pool, k=2)
            
            random_selected_results = []

            for randomly_selected_id_n_tag in randomly_selected_id_n_tags:

                block_id = randomly_selected_id_n_tag[0]
                html_tag = randomly_selected_id_n_tag[1]
 

                random_selected_dict = {
                    'Section_name': section_name,
                    'Block_id': block_id,
                    'HTML_Tag': html_tag
                }
                random_selected_results.append(random_selected_dict)
                
            return random_selected_results
        # NOTE 250422 : 같은 방식이지만 혹시를 대비하여 
        except Exception as e:
            
            print(f"[DEBUG] random.sample() was not working. Maybe the block recommendation is just 1. {e}")

            random_index = random.randint(0, len(block_list[1].keys()) - 1)
            
            block_keys = list(block_list[1].keys())
            selected_block_id = block_keys[random_index]
            selected_html_tag = tag_slice[random_index] if len(tag_slice) > random_index else tag_slice[0]  # tag_slice 길이 확인

            return [{
                'Section_name': section_name,
                'Block_id': selected_block_id,
                'HTML_Tag': selected_html_tag
            }]


    async def select_block_batch_randomly(
        self,
        section_names_n_block_lists: List[Dict[str, str]]) -> List[Dict[str, Any]]:

        # NOTE 250424 : 우선 한 번 더 감싸는거 배제함. 추후 batch에서 문제 생기면 복원 예정
        select_block_results = []
        for block_list in section_names_n_block_lists:

            temp_block_list = list(block_list.items())

            for n_temp_block_list in temp_block_list:
                
                select_block_result = await self.select_block_randomly(n_temp_block_list)
                select_block_results.append(select_block_result)

        # NOTE 250220: batch 방식은 추후 적용        
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


# class OpenAIBlockSelector:
#     def __init__(self, batch_handler):
#         self.batch_handler = batch_handler