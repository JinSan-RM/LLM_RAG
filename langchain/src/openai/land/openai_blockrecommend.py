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
        

        # 원래 section_name 인데 이걸 section_context로로 변경
        # 왜냐면 이미 1차적으로 섹션이 정해진 것이기 떄문에, 컨텐츠를 잘 표현할 리스트를 찾는게 맞음
        prompt = f"""
                [System]
                You are an AI assistant that helps you select HTML tags. Please follow these instructions.
                
                #### Explain Semantic Tag ####
                h1: Represents the most important title of the web page. Typically only one is used per page, and it represents the topic or purpose of the page.
                h2: Indicates the next most important heading after h1. It is mainly used to separate major sections of a page.
                h3: A subheading of h2, indicating detailed topics within the section.
                h5: It is located around h tags and briefly assists them.
                p: A tag that defines a paragraph. Used to group plain text content.
                li: Represents a list item. Mainly used within ul (unordered list) or ol (ordered list) tags.
                
                #### INSTRUCTIONS ####
                1. READ THE section context FROM USER.
                2. READ Explain Semantic Tag ABOVE.
                2. SELECT ONLY ONE TAG FROM THE tag list THAT CAN BEST EXPRESS THE section context.
                3. RETURN ONLY ONE TAG. NEVER ADD THE OTHER WORDS.
                4. KEEP THE STRUCTURE GIVEN.
                5. DO NOT PRINT ANY OTHER TEXT.
                6. BE SURE TO SELECT ONLY ONE OF THE DATA IN THE TAG LIST TO PRINT.

                
                [/System]
                
                [Example_User]
                User input:
                section context = "제일철강은 고품질 철강 제조 및 판매를 주력으로 하는 기업으로서, 철강 제련 및 유통 사업을 전개하고자 합니다. 당사는 제조업체와 건설회사를 1차 목표 시장으로 설정하고, 중소 철강 가공업체로 시장을 확대하며, 장기적으로는 수출 시장 진출을 목표로 하고 있습니다."
                tag list = ["h1_h2_p", "li(h2+h5+p)*2", "h5_h1_p"]
                [/Example_User]
                
                [Example_Assistant]
                Example_Output:
                {{"selected_tag": "li(h2+h5+p)*2"}}
                [/Example_Assistant]

                [User]
                User input:
                section context = {only_section_context}
                tag list = {tag_slice}
                [/User]
                """
        
        result = await self.send_request(prompt, max_tokens)
        
        # NOTE 250219 : 기존 형식 맞추기위해서 다시 dict 형식으로 전환
        # 추후에는 find_key_by_value()과 extract_emmet_tag의 동작 방식 전환
        block_list_dict = {block_list[0]:block_list[1]}
        # print("check check : ", block_list)

        # NOTE 250219 : 아래 함수를 사용하기 위해서 extract_json
        if result.data.generations[0][0].text.strip() == None:
            selected_Html_tag_str = tag_slice[0]
        else:
            selected_Html_tag_json = self.extract_json(result.data.generations[0][0].text.strip())
            selected_Html_tag_str = str(list(selected_Html_tag_json.values())[0])
        
        if selected_Html_tag_str not in tag_slice:
            selected_Html_tag_str = tag_slice[0]
        # NOTE 250219 : 여기는 좀 나중에 보자... 눈 빠지것다
        #               지금은 임시고, 여기 검증단계 살려야합니다아아
        # b_id = self.find_key_by_value(block_list_dict, selected_Html_tag_str)
        # if b_id is None:
        #     raise ValueError(f"매칭되는 b_id가 없습니다: ")
        
        # b_value = self.extract_emmet_tag(block_list_dict[b_id])
        
        section_name = section_context[0]
        reversed_block_dict = {value: key for key, value in block_list[1].items()}
        
        # NOTE 250219 : 정동이사님 요청으로 Section_name 함께 반환
        result.data.generations[0][0].text = {
            'Section_name': section_name,
            'Block_id': reversed_block_dict[selected_Html_tag_str], # b_id
            'HTML_Tag': selected_Html_tag_str # b_value
        }
        
        # print("++++++++++++++++++++++++++++++++++++++++")
        # print("result : ", result)
        
        return result

    async def select_block_batch(
        self,
        section_names_n_contents: List[str],
        section_names_n_block_lists: List[Dict[str, str]],
        max_tokens: int = 50) -> List[Dict[str, Any]]:
        
        # print("+++++++++++++++++++++++++++++++++")
        # print("type(section_names_n_contents) : ", type(section_names_n_contents))
        # print("section_names_n_contents : ", section_names_n_contents)
        # print("section_names_n_contents[0] : ", section_names_n_contents[0])
        # print("type(section_names_n_block_lists) : ", type(section_names_n_block_lists))
        # print("section_names_n_block_lists : ", section_names_n_block_lists)
        # print("section_names_n_block_lists[0] : ", section_names_n_block_lists[0])
    
        select_block_results = []
        for section_name, block_list in zip(section_names_n_contents, section_names_n_block_lists):
            
            # print("++++++++++==========++++++++++")
            # print("type(section_name) : ", type(section_name))
            # print("section_name : ", section_name)
            # print("type(block_list) : ", type(block_list))
            # print("block_list : ", block_list)
            
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
