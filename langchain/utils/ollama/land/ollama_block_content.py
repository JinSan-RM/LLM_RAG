from config.config import OLLAMA_API_URL
import requests, json, re


class OllamaBlockContent:
    
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.4, model:str = ''):
        self.api_url = api_url
        self.temperature = temperature
        self.model = model
    
    async def send_request(self, prompt: str) -> str:
        """
        공통 요청 처리 함수 : API 호출 및 응답 처리
        
        Generate 버전전
        """
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
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

        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 실패: {e}")
            raise RuntimeError(f"Ollama API 요청 실패: {e}")
        
    async def generate_block_content(self, summary:str, section_name: str, HTMLtag: str ):
        """
        랜딩 페이지 섹션을 생성하는 함수
        """

        # 블록태그를 풀어서 html tag구조를 만드는 알고리즘 필요.
        # html tag태그 구조로 summary 데이터를 기준으로 데이터 만들기.
        prompt=f"""
        """
        repeat_count = 0
        while repeat_count < 3:
            try:
                # 1) LLM에 요청
                print(f"len prompt : {len(prompt)}")
                raw_json = await self.send_request(prompt=prompt)

                raw_json = re.sub("\n", "", raw_json)
                print(f"raw_json : {type(raw_json)} {len(raw_json)} / {raw_json}")


                return 
            except RuntimeError as r:
                print(f"Runtime error: {r}")
                repeat_count += 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                repeat_count += 1
        raise RuntimeError("Failed to parse JSON after 3 attempts")