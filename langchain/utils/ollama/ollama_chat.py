import requests, json, random, re
from config.config import OLLAMA_API_URL
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


class OllamaChatClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/chat', temperature=0.4, structure_limit = True,  n_ctx = 4196, max_token = 3000):
        self.api_url = api_url
        self.temperature = temperature
        self.structure_limit = structure_limit
        self.n_ctx = n_ctx
        self.max_token = max_token
        self.message_history = []
        
    async def send_request(self, model: str, messages: list) -> str:
        """
        공통 요청 처리 함수: /chat API 호출 및 응답 처리
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "n_ctx": self.n_ctx,
            "repetition_penalty": 1.2,
            "session": "test_session",
            "stream" : False
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리

            response_json = response.json()
            print(f"response_json : {response_json}")
            assistant_reply = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return assistant_reply.strip() if assistant_reply else "Empty response received"

        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 실패: {e}")
            raise RuntimeError(f"Ollama API 요청 실패: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return "Invalid JSON response received"
        
    def split_into_chunks(self, data: str, max_tokens: int) -> list:
        """
        데이터를 최대 토큰 수에 맞게 청크로 분할하는 함수

        :param data: 입력 데이터 문자열
        :param max_tokens: 각 청크의 최대 토큰 수
        :return: 분할된 청크 리스트
        """
        paragraphs = data.split('\n\n')
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk.split()) + len(para.split()) <= max_tokens:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    async def store_chunks(self, model: str, data: str):
        """
        대용량 데이터를 청크로 분할하고, 각 청크를 모델에 전달하여 컨텍스트로 저장하는 함수

        :param model: 사용하려는 모델 이름
        :param data: 대용량 입력 데이터 문자열
        """
        max_tokens_per_chunk = 2500  # 각 청크의 최대 토큰 수 (예시)
        chunks = self.split_into_chunks(data, max_tokens_per_chunk)

        for chunk in chunks:
            # 메시지 형식으로 변환
            user_message = {"role":"system", 
                            "content":"내가 입력한 데이터들을 기억하고있구 입력된 데이터를 기준으로 출력을 해야할 때, 이전 입력한 데이터를 기반으로 작성해줘.",
                            "role": "user",
                            "content": f"{chunk}"}

            # API 요청 (비동기 함수이므로 await 사용)
            response = await self.send_request(model, [user_message])

            # 필요에 따라 응답을 처리할 수 있습니다.
            print(f"Chunk response: {response}")

    async def generate_section(self, model: str, section_name: str) -> str:
        """
        특정 섹션의 랜딩 페이지 콘텐츠를 생성하는 함수

        :param model: 사용하려는 모델 이름
        :param section_name: 생성하려는 섹션의 이름
        :return: 생성된 섹션의 콘텐츠
        """
        # 시스템 메시지 설정
        message = {
            "role": "system",
            "content": f"""
            - 너는 사이트의 섹션 구조를 정해주고, 그 안에 들어갈 내용을 작성해주는 AI 도우미야.
            - 입력된 데이터를 기준으로 단일 페이지를 갖는 랜딩사이트 콘텐츠를 생성해야 해.
            - 'children'의 컨텐츠 내용의 수는 너가 생각하기에 섹션에 알맞게 개수를 수정해서 생성해줘.
            - 섹션 '{section_name}'에 어울리는 내용을 생성해야 하며, 반드시 다음 규칙을 따라야 한다:
                1. assistant처럼 생성해야 하고 형식을 **절대** 벗어나면 안 된다.
                2. "div, h1, h2, h3, p, ul, li" 태그만 사용해서 섹션의 콘텐츠를 구성해라.
                3. 섹션 안의 `children` 안의 컨텐츠 개수는 2~10개 사이에서 자유롭게 선택하되, 내용이 반복되지 않도록 다양하게 생성하라.
                4. 모든 텍스트 내용은 입력 데이터에 맞게 작성하고, 섹션의 목적과 흐름에 맞춰야 한다.
                5. 출력 결과는 코드 형태만 허용된다. 코드는 **절대 생성하지 마라.**
                6. 오직 한글로만 작성하라.
            """,
            "role": "user",
            "content":
            f"""
            입력 데이터:
            내가 입력했던 section에 알맞는 내용 생성해줘.
            섹션:
            {section_name}
            """
        }

        response_user = await self.send_request(model, [message])
        print(f"User response: {response_user}")
        

        return response_user.strip()