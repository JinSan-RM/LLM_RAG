import requests
import json
from config.config import OLLAMA_API_URL


class OllamaUsrMsgClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.4, usr_msg: str = '', model: str = ''):
        self.api_url = api_url
        self.temperature = temperature
        self.usr_msg = usr_msg
        self.model = model

    async def send_request(self, prompt: str) -> str:
        """
        공통 요청 처리 함수: /chat API 호출 및 응답 처리
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리

            full_response = response.text  # 전체 응답
            lines = full_response.splitlines()
            all_text = ""
            for line in lines:
                try:
                    json_line = json.loads(line.strip())  # 각 줄을 JSON 파싱
                    all_text += json_line.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    continue  # JSON 파싱 오류 시 건너뛰기

            return all_text.strip() if all_text else "Empty response received"

        except requests.exceptions.Timeout:
            print("The request timed out.")
        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 실패: {e}")
            raise RuntimeError(f"Ollama API 요청 실패: {e}") from e

    async def usr_msg_process(self):
        prompt = f"""
        <|start_header_id|>system<|end_header_id|>
        - 당신은 전문 웹사이트 랜딩 페이지를 만드는 디자이너이자 카피라이터입니다.
        - 단순히 항목별로 포맷을 나누는 것이 아니라, 입력된 데이터를 하나의 자연스러운 컨텍스트 내에서 사업 계획서처럼 풀어내어 서술해야 합니다.
        - 브랜드의 정체성, 가치, 경쟁력, 고객의 문제와 그 해결책, 그리고 실행 전략 등을 하나의 이야기를 구성하듯이 전개해 주세요.
        - 문장은 전문적이면서도 읽기 쉽게 자연스러운 흐름으로 작성되어야 하며, 사업의 비전과 전략이 잘 드러나도록 서술해 주세요.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        입력 데이터:
        {self.usr_msg}

        <|start_header_id|>assistant<|end_header_id|>
        입력 데이터를 바탕으로, 브랜드의 정체성과 가치를 효과적으로 전달하면서도 방문자가 흥미를 느낄 수 있도록, 하나의 연속된 맥락 안에서 사업 계획서 느낌의 콘텐츠를 작성해 주세요. 브랜드의 배경, 비전, 경쟁력, 고객 문제 및 해결책, 그리고 향후 전략과 기대 효과 등을 자연스럽게 녹여내어 서술해 주시기 바랍니다.
        """
        usr_data = await self.send_request(prompt=prompt)
        return usr_data