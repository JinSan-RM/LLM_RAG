from langchain.prompts import PromptTemplate
import asyncio
from vllm import SamplingParams

class OpenAIDataMergeClient:
    def __init__(self, usr_msg: str, pdf_data: str, batch_handler):
        self.usr_msg = usr_msg
        self.pdf_data = pdf_data
        self.batch_handler = batch_handler

    async def contents_merge(self) -> str:
        try:
            prompt=f"""
            System: 기업 사이트에 들어갈 개요를 만들어주는 전문가야.

            아래 순서에 따라서 작업을 진행해줘.

            1. 사용자 입력이 들어오면, 그걸 PDF 요약 데이터보다 강조해서 작업을 진행해줘.

            2. 사용자 입력와 PDF 요약 데이터를 살펴보면 아래 8가지 정보들이 포함되어 있을거야.
            만약 모자란 정보가 있다면 PDF 요약 데이터를 함께 고려하면서 너가 적당한 창의성을 발휘해서 작성해줘.

                1) 사업 아이템 : 구체적인 제품 또는 서비스 내용
                2) 슬로건 혹은 캐치 프레이즈 : 기업의 주요 비전이나 이념을 한 문장으로 표현
                3) 타겟 고객 : 주요 고객층의 특성과 니즈
                4) 핵심 가치 제안 : 고객에게 제공하는 독특한 가치
                5) 제품 및 서비스 특징 : 주요 기능과 장점
                6) 비즈니스 모델 : 잠재 고객에게 제품이나 서비스에 차별화된 가치를 담아 제공함으로써 수익을 창출하는 일련의 프로세스
                7) 홍보 및 마케팅 전략 : 제품이나 서비스를 고객에게 소개하는 방법

            3. 최종 출력은 String 형식 이외에 어떤 설명, 문장, 주석도 작성하지 마.

            사용자 입력 :
            {self.usr_msg}

            PDF 요약 데이터 :
            {self.pdf_data}

            사용자 입력을 우선적으로 포함하여 내용을 작성하고 나머지 내용은 PDF 요약데이터로 작성해줘.
            내용을 작성할 땐, 두개가 고루 혼합되어 작성 되었으면 좋겠어.
            """
            sampling_params = SamplingParams(max_tokens=2000)
            request = {
                "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
                "sampling_params": sampling_params,
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.1,
                "top_p": 0.8
            }
            result = await asyncio.wait_for(
            self.batch_handler.process_single_request(request, 0),
            timeout=120
            )

            if result.success:
                response_text = result.data.generations[0][0].text.strip()
                print(f"merge response_text : {response_text}")
                response_text = str(response_text)
                return response_text  # 생성된 텍스트를 직접 반환

            else:
                print(f"Error in contents merge: {result.error}")
                return ""
        except asyncio.TimeoutError:
            print("Contents merge request timed out")
            return ""

