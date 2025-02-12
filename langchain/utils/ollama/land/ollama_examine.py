from config.config import OLLAMA_API_URL
import requests
import json
import aiohttp
import asyncio


class OllamaExamineClient:

    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.3, model: str = '', data: str = ''):
        self.api_url = api_url
        self.temperature = temperature
        self.model = model
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


    async def data_examine(self) -> str:
        """
        데이터 검토 메서드: 입력 데이터에서 비속어 검출
        """
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        너는 비속어, 욕을 검열한 뒤, 검열이 되면 "비속어"만 출력하는 AI야.

        작업 순서:
        1. 입력 데이터를 확인한다.
        2. 비속어 또는 욕설이 포함된 경우, '비속어'라는 단어만 반환한다.
        3. 출력은 반드시 String 형식만 포함한다.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {self.data}

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        return await self.send_request(prompt=prompt)
