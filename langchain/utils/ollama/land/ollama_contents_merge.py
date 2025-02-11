from config.config import OLLAMA_API_URL
import requests
import json
import aiohttp
import asyncio


class OllamaDataMergeClient:

    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.7, model: str = '', user_msg: str = '', data: str = ''):
        self.api_url = api_url
        self.temperature = temperature
        self.model = model
        self.user_msg = user_msg
        self.data = data

    async def send_request(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
        }
        # aiohttp ClientSession을 사용하여 비동기 HTTP 요청 수행
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.api_url, json=payload, timeout=15) as response:
                    response.raise_for_status()  # HTTP 에러 발생 시 예외 처리
                    full_response = await response.text()  # 응답을 비동기적으로 읽기
            except aiohttp.ClientError as e:
                print(f"HTTP 요청 실패: {e}")
                raise RuntimeError(f"Ollama API 요청 실패: {e}") from e

        # 전체 응답을 줄 단위로 분할하고 JSON 파싱
        lines = full_response.splitlines()
        all_text = ""
        for line in lines:
            try:
                json_line = json.loads(line.strip())
                all_text += json_line.get("response", "")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue

        return all_text.strip() if all_text else "Empty response received"

    async def contents_merge(self) -> str:
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>

        기업 사이트에 들어갈 개요를 만들어주는 전문가야.

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

        3. 최종 출력은 **String** 형식 이외에 어떤 설명, 문장, 주석도 작성하지 마.

        <|eot_id|><|start_header_id|>user<|end_header_id|>

        사용자 입력 :
        {self.user_msg}

        PDF 요약 데이터 :
        {self.data}

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        사용자 입력을 우선적으로 포함하여 내용을 작성하고 나머지 내용은 PDF 요약데이터로 작성해줘.
        내용을 작성할 땐, 두개가 고루 혼합되어 작성 되었으면 좋겠어.
        """
        result = await self.send_request(prompt=prompt)
        print(result, "<=====result")
        return result
