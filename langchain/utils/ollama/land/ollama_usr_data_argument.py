import requests
import json
from config.config import OLLAMA_API_URL
import aiohttp
import asyncio


class OllamaUsrMsgClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.4, usr_msg: str = '', model: str = ''):
        self.api_url = api_url
        self.temperature = temperature
        self.usr_msg = usr_msg
        self.model = model
    
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

    async def usr_msg_process(self):
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        당신은 다음과 같은 전문성을 가진 AI 어시스턴트입니다:
        - 웹사이트 랜딩 페이지 디자인 전문가
        - 전문 카피라이터
        - 사업계획서 작성 전문가

        작업 요구사항
        1. 입력된 데이터를 바탕으로 사업계획서 스타일의 내러티브 콘텐츠를 작성하되:
        - 형식에 얽매이지 않는 자유로운 서술
        - 전체 글자 수 500자 이하 유지
                
        콘텐츠 품질 기준:
        - 전문적인 어조 유지
        - 이해하기 쉬운 설명
        - 논리적인 흐름
        - 설득력 있는 서술
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {self.usr_msg}
        <|start_header_id|>assistant<|end_header_id|>
        # 출력 형식
        1. 서술형 텍스트로 작성
        2. 별도의 섹션 구분 없이 자연스러운 흐름
        """
        print(f"usr_msg prompt : {prompt}")
        usr_data = await self.send_request(prompt=prompt)
        return usr_data