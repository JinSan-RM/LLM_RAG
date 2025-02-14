from langchain.prompts import PromptTemplate
import asyncio

class OpenAIUsrMsgClient:
    def __init__(self,  usr_msg: str, batch_handler):
        self.batch_handler = batch_handler
        self.usr_msg = usr_msg
        self.prompt_template = PromptTemplate(
            input_variables=["usr_msg"],
            template=f"""
            당신은 웹사이트 랜딩 페이지 디자인 전문가, 전문 카피라이터, 사업계획서 작성 전문가입니다. 다음 요구사항에 따라 사업계획서 스타일의 내러티브 콘텐츠를 작성하세요:
            - 형식에 얽매이지 않는 자유로운 서술
            - 전체 글자 수 500자 이하 유지
            - 전문적인 어조 유지
            - 이해하기 쉬운 설명
            - 논리적인 흐름
            - 설득력 있는 서술

            입력 데이터:
            {usr_msg}

            출력 형식:
            1. 서술형 텍스트로 작성
            2. 별도의 섹션 구분 없이 자연스러운 흐름
            """
        )

    async def usr_msg_proposal(self):
        try:
            prompt = self.prompt_template.format(user_input=self.usr_msg)
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "n": 1,
                    "stream": False,
                    "logprobs": None
                }, request_id=0),
                timeout=30  # 적절한 타임아웃 값 설정
            )
            return response
        except asyncio.TimeoutError:
            print("User message proposal request timed out")
            return None