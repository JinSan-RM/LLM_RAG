from langchain.prompts import PromptTemplate
import asyncio

class OpenAIDataMergeClient:
    def __init__(self, usr_msg: str, pdf_data: str, batch_handler):
        
        self.usr_msg = self.extract_text(usr_msg)
        self.pdf_data = self.extract_text(pdf_data)
        self.batch_handler = batch_handler

    async def contents_merge(self) -> str:
        try:
            
            prompt = f"""
            <|start_header_id|>SYSTEM<|end_header_id|>
            you are an expert in writing business plans.
            Write a business plan using the user summary and PDF summary data.
            
            #### INSTRUCTIONS ####

            1. WHEN user summary COMES IN, PROCEED WITH THE TASK BY EMPHASIZING IT OVER THE pdf summary data.
            2. FOLLOW THE INPUT LANGUAGE TO THE OUTPUT.
            3. DATA WILL CONTAIN THE SEVEN PIECES OF INFORMATION BELOW.
            IF THERE IS INSUFFICIENT INFORMATION, PLEASE FILL IT OUT USING APPROPRIATE CREATIVITY WITH user summary and THE pdf summary data.

                1) BUSINESS ITEM: SPECIFIC PRODUCT OR SERVICE DETAILS
                2) SLOGAN OR CATCH PHRASE: EXPRESSING THE COMPANY'S MAIN VISION OR IDEOLOGY IN ONE SENTENCE
                3) TARGET CUSTOMERS: CHARACTERISTICS AND NEEDS OF MAJOR CUSTOMER BASES
                4) CORE VALUE PROPOSITION: UNIQUE VALUE PROVIDED TO CUSTOMERS
                5) PRODUCT AND SERVICE FEATURES: MAIN FUNCTIONS AND ADVANTAGES
                6) BUSINESS MODEL: A SERIES OF PROCESSES THAT GENERATE PROFITS BY PROVIDING DIFFERENTIATED VALUE IN PRODUCTS OR SERVICES TO POTENTIAL CUSTOMERS.
                7) PROMOTION AND MARKETING STRATEGY: HOW TO INTRODUCE PRODUCTS OR SERVICES TO CUSTOMERS
                
            4. DO NOT WRITE ANY EXPRESSION OTHER THAN THE FINAL OUTPUT.
            5. PLEASE WRITE ABOUT 500 TO 1000 CHARACTERS SO THAT THE CONTENT IS RICH AND CONVEYS THE CONTENT.
            6. WHEN WRITING CONTENT, MIX USER INPUT AND PDF INPUT EVENLY.
            7. NEVER WRITE SYSTEM PROMPT LIKE THESE <|start_header_id|>SYSTEM<|end_header_id|>" IN THE OUPUT.
            
            <|eot_id|><|start_header_id|>USER_EXAMPLE<|end_header_id|>
            user summary = "data from user"
            pdf summary data = "data from pdf"
            
            <|eot_id|><|start_header_id|>ASSISTANT_EXAMPLE<|end_header_id|>
            {{Output : "narrative summary of merged data"}}
            
            <|eot_id|><|start_header_id|>USER<|end_header_id|>
            user summary = {self.usr_msg}
            pdf summary data = {self.pdf_data}
            
            """
            print(f"usr_msg : {self.usr_msg}")
            print(f"pdf_data : {self.pdf_data}")
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                    "prompt": prompt,
                    "max_tokens": 2000,
                    "temperature": 0.6,
                    "top_p": 0.1,
                    "repetition_penalty" : 1.5,
                    "frequency_penalty" : 1.5,
                    "n": 1,
                    "stream": False,
                    "logprobs": None
                }, request_id=0),
                timeout=60  # 적절한 타임아웃 값 설정
            )
            return response
        except asyncio.TimeoutError:
            print("Contents merge request timed out")
            return ""

            # NOTE 250219 : 여기도 send_request() 방식으로 바꿔서 [System] 같은 글자 안나오게 만들기
            #               굳이  적용을 안해도 response로 쏙 들어가서 결과값 치환해줘도 되겠군
            # selected_usr_result = self.extract_json(response.data.generations[0][0].text.strip())
            # # selected_usr_result_str = str(list(selected_usr_result.values())[0])            



    def extract_text(self, result):
        if result.success and result.data.generations:
            return result.data.generations[0][0].text
        else:
            return "텍스트 생성 실패"
        
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