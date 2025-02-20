from langchain.prompts import PromptTemplate
import asyncio
import re, json
class OpenAIUsrMsgClient:
    def __init__(self,  usr_msg: str, batch_handler):
        self.batch_handler = batch_handler
        self.usr_msg = usr_msg
        # self.prompt_template = PromptTemplate(
        #     input_variables=["usr_msg"],
        #     template=f"""
        #     You are a professional in business plan writing. You are provided with an input in form of sentence or paragraph. Your task is to provide a paragraph between 300-700 characters for assisting write business plan.

        #     #### Instructions ####

        #     Step 1. Read the input. The basic language is Korean. But if provided in other language, use it. Fix any typos. 
        #     Step 2. If input has important information like "Company name, Brand name, pros and corps, target customers", must include that in the paragraph for keywords.
        #     Step 3. Think about the keywords step by step and write out your business plan narratively. Limit topic to only those provided in input without changing the scope.
        #     Step 4. Format the output as valid string.
        #     Step 5. The total output must be more than 2000 characters.
            
        #     #### Example 1 ####
        #     Input:
        #     "철을 제련해서 파는 사업을 하려고 해. 회사 이름은 '제일철강'이야. 우리는 철을 필요로 하는 모든 사람들에게 높은 품질의 철을 판매할 예정이야."

        #     Output:
        #     "사업계획": "제일철강은 고품질 철강 제조 및 판매를 주력으로 하는 기업으로서, 철강 제련 및 유통 사업을 전개하고자 합니다. 당사는 제조업체와 건설회사를 1차 목표 시장으로 설정하고, 중소 철강 가공업체로 시장을 확대하며, 장기적으로는 수출 시장 진출을 목표로 하고 있습니다. 철저한 품질 관리 시스템과 안정적인 원자재 수급망을 바탕으로 원가 경쟁력을 확보하고, 산업별 맞춤형 제품 공급과 신속한 납기 대응, 기술 지원 서비스를 제공할 계획입니다. 이를 통해 철강 산업의 신뢰할 수 있는 파트너로 자리매김하여, 고품질 제품과 전문적인 서비스로 고객 만족을 실현하겠습니다. 당사는 고품질 철강의 안정적 생산 및 공급을 통해 다양한 산업 분야의 수요를 충족시키며, 철강 시장에서 신뢰성 있는 공급자로서의 위치를 확고히 하겠습니다."
        #     """
        # )

    async def usr_msg_proposal(self):
        try:
            # prompt = self.prompt_template.format(user_input=self.usr_msg)
            prompt = f"""
            [System]
            You are a professional in business plan writing. 
            You are provided with an user input in form of sentence or paragraph. 
            Your task is to provide a paragraph for assisting write business plan.

            #### INSTRUCTIONS ####

            STEP 1. READ THE USER INPUT. THE BASIC LANGUAGE IS KOREAN. BUT IF PROVIDED IN OTHER LANGUAGE, SELECT IT FOR OUTPUT.
            STEP 2. IF USER INPUT HAS IMPORTANT INFORMATION LIKE "COMPANY NAME, BRAND NAME, PROS AND CORPS, TARGET CUSTOMERS", MUST INCLUDE THAT IN THE PARAGRAPH FOR KEYWORDS.
            STEP 3. THINK ABOUT THE KEYWORDS STEP BY STEP AND WRITE OUT YOUR BUSINESS PLAN NARRATIVELY. LIMIT TOPIC TO ONLY THOSE PROVIDED IN USER INPUT WITHOUT CHANGING THE SCOPE.
            STEP 4. PLEASE WRITE ABOUT 1000 TO 1500 CHARACTERS SO THAT THE CONTENT IS RICH AND CONVEYS THE CONTENT.
            STEP 5. NEVER REPEAT SAME CONTENT FOR FITTING THE OUTPUT CHARACTERS. USE THE OTHER WORDS AND EXPRESSIONS.
            STEP 6. NEVER REPEAT User-Assistant FOR FITTING THE OUTPUT CHARACTERS. JUST END GENERATE.
            STEP 7. FIX ANY TYPOS. 
            
            
            #### Output Format Example ####
            "Write business plan, narratively"
            [/System]

            [User_Example]
            user input:
            "철을 제련해서 파는 사업을 하려고 해. 회사 이름은 '제일철강'이야. 우리는 철을 필요로 하는 모든 사람들에게 높은 품질의 철을 판매할 예정이야."
            [/User_Example]
            
            [Assistant_Example]
            Output:
            "제일철강은 고품질 철강 제조 및 판매를 주력으로 하는 기업으로서, 철강 제련 및 유통 사업을 전개하고자 합니다. 당사는 제조업체와 건설회사를 1차 목표 시장으로 설정하고, 중소 철강 가공업체로 시장을 확대하며, 장기적으로는 수출 시장 진출을 목표로 하고 있습니다. 철저한 품질 관리 시스템과 안정적인 원자재 수급망을 바탕으로 원가 경쟁력을 확보하고, 산업별 맞춤형 제품 공급과 신속한 납기 대응, 기술 지원 서비스를 제공할 계획입니다. 이를 통해 철강 산업의 신뢰할 수 있는 파트너로 자리매김하여, 고품질 제품과 전문적인 서비스로 고객 만족을 실현하겠습니다. 당사는 고품질 철강의 안정적 생산 및 공급을 통해 다양한 산업 분야의 수요를 충족시키며, 철강 시장에서 신뢰성 있는 공급자로서의 위치를 확고히 하겠습니다."
            [/Assistant_Example]
            
            [User]
            user input = {self.usr_msg}
            [/User]
            """
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                    "prompt": prompt,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "top_p": 0.3,
                    "repetition_penalty" : 1.5,
                    "n": 1,
                    "stream": False,
                    "logprobs": None
                }, request_id=0),
                timeout=60  # 적절한 타임아웃 값 설정
            )
            
            # NOTE 250219 : 여기도 send_request() 방식으로 바꿔서 [System] 같은 글자 안나오게 만들기
            #               굳이  적용을 안해도 response로 쏙 들어가서 결과값 치환해줘도 되겠군
            # selected_usr_result = self.extract_json(response.data.generations[0][0].text.strip())
            # # selected_usr_result_str = str(list(selected_usr_result.values())[0])            
            
            
            return response
        except asyncio.TimeoutError:
            print("User message proposal request timed out")
            return None
        
    # def extract_json(self, text):
    # # 가장 바깥쪽의 중괄호 쌍을 찾습니다.
    #     text = re.sub(r'[\n\r\\\\/]', '', text, flags=re.DOTALL)
    #     json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', text, re.DOTALL)
    #     if json_match:
    #         json_str = json_match.group()
    #     else:
    #         # Handle case where only opening brace is found
    #         json_str = re.search(r'\{.*', text, re.DOTALL)
    #         if json_str:
    #             json_str = json_str.group() + '}'
    #         else:
    #             return None

    #     # Balance braces if necessary
    #     open_braces = json_str.count('{')
    #     close_braces = json_str.count('}')
    #     if open_braces > close_braces:
    #         json_str += '}' * (open_braces - close_braces)

    #     try:
    #         return json.loads(json_str)
    #     except json.JSONDecodeError:
    #         # Try a more lenient parsing approach
    #         try:
    #             return json.loads(json_str.replace("'", '"'))
    #         except json.JSONDecodeError:
    #             return None        