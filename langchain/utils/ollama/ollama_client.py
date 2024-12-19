
import requests
import json
from typing import Optional, List, Any
from langchain.llms.base import BaseLLM
from config.config import OLLAMA_API_URL
from langchain.schema import LLMResult
from pydantic import Field
from script.prompt import MENU_STRUCTURE, TITLE_STRUCTURE, KEYWORDS_STRUCTURE
import re
class OllamaClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.5):
        self.api_url = api_url
        self.temperature = temperature

        
        
    def generate(self, model: str, prompt: str) -> str:
        """
        Ollama API를 사용하여 텍스트 생성
        Args:
            model (str): 사용할 Ollama 모델 이름
            prompt (str): 입력 프롬프트

        Returns:
            str: Ollama 모델의 생성된 텍스트
        """
        payload = {
            "model": model,
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
  
    def PDF_Menu(self, model: str, text: str) -> dict:
        """
        Ollama API를 사용하여 텍스트 생성
        Args:
            model (str): 사용할 Ollama 모델 이름
            prompt (str): 입력 프롬프트

        Returns:
            str: Ollama 모델의 생성된 텍스트
        """
        # 요청 페이로드 구성
        # prompt = f"""
        #         <|start_header_id|>system<|end_header_id|>
        #         - You are the organizer of the company introduction website. 
        #         - Data is a company profile or company introduction. 
        #         - The result values will be printed in three ways.
        #         - Data must be generated based absolutely on the sample structure below.
                
        #         1. "title_structure": the title of the website.
        #         2. "keywords_structure": a list of 3 keywords extracted from the content.
        #         3. "menu_structure": two_depth menu structure. first_depth should be 3-5 and second_depth should be 0-4. No further description is required for second_depth. Menu items should be less than 15 characters long.
        #         - Answer only with the JSON object without additional text.

                
        #         <|eot_id|><|start_header_id|>user<|end_header_id|>
        #         Input data:
        #         {text}

        #         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        #         **Example Structures:**

        #         {TITLE_STRUCTURE}

        #         {KEYWORDS_STRUCTURE}

        #         {MENU_STRUCTURE}

        #         {{"title_structure": "The Example of Company Introduction",
        #                 "keywords_structure": ["Company", "Introduction", "Jellywork"],
        #                 "menu_structure": [
        #                     "1. Home, ",
        #                     "2. Company Introduction, ",
        #                         "- Company History",
        #                         "- Company Vision",
        #                         "- CEO Message",
        #                     "3. Business Overview, ",
        #                         "- Business Areas",
        #                         "- Business Achievements",
        #                         "- Future Goals",
        #                     "4. Contact Us, ",
        #                         "- Location",
        #                         "- Phone",
        #                         "- FAQs",
        #                         "- Team members"
        #                 ]
        #             }}

        #         """
        
        prompt = f"""
               <|start_header_id|>system<|end_header_id|>
                - 당신은 회사 소개 웹사이트의 기획자입니다.
                - Data는 회사 프로필 또는 회사 소개입니다.
                - 결과 값은 세 가지 방식으로 출력됩니다.
                - Data는 아래 샘플 구조를 절대적으로 기반으로 생성되어야 합니다.
                
                1. "title_structure": 입력데이터에서 추출한 웹사이트 제목입니다.
                2. "keywords_structure": 입력데이터에서 추출한 3개의 키워드 리스트입니다.
                3. "menu_structure": 입력데이터를 기반으로 이중 메뉴 구조로, first_depth는 3~5개, second_depth는 0~4개로 작성됩니다. second_depth에 대한 추가 설명은 필요하지 않습니다. 메뉴 항목은 15자 이내로 작성되어야 합니다.
                - 추가 텍스트 없이 JSON 객체로만 답변하십시오.
                - 절대 code 데이터를 생성하지 마세요.

                
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                입력 데이터:
                {text}

                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                **샘플 구조:**

                {TITLE_STRUCTURE}

                {KEYWORDS_STRUCTURE}

                {MENU_STRUCTURE}
                {{"title_structure": "회사 소개페이지",
                        "keywords_structure": ["회사", "소개", "사업아이템"],
                        "menu_structure": [
                            "1. 홈, ",
                            "2. 회사 소개, ",
                                "- 회사 역사",
                                "- 회사 비전",
                                "- CEO 메시지",
                            "3. 사업 개요, ",
                                "- 사업 영역",
                                "- 사업 실적",
                                "- 미래 목표",
                            "4. 문의하기, ",
                                "- 위치",
                                "- 전화",
                                "- 자주 묻는 질문",
                                "- 팀 멤버"
                        ]
                    }}
        """
        payload = {
            "model": model,  # 사용 중인 Ollama 모델 이름으로 변경하세요
            "prompt": prompt,
            "temperature": 0.1,
            # "top_k": 0.1,
            # "top_p": 0.25
        }
        try:
            print("start response : ", len(prompt))
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
                
            all_text = parse_response(all_text)
            all_text = all_text[0]
            print("all_text :", all_text[0])
            return all_text.strip() if all_text else "Empty response received"

        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 실패: {e}")
            raise RuntimeError(f"Ollama API 요청 실패: {e}")
        
    def PDF_Menu_Contents(self, model: str, text: str, menu: str):
        """
        Ollama API를 사용하여 텍스트 생성
        Args:
            model (str): 사용할 Ollama 모델 이름
            prompt (str): 입력 프롬프트
            menu (str): 사용할 메뉴의 이름

        Returns:
            str: Ollama 모델의 생성된 텍스트
        """
        
        prompt = f"""0. 너는 각 메뉴에 알맞는 컨텐츠를 작성해 줄 거야. 참고할 자료는 {menu}와 {text}이야
                1. 먼저 "메뉴"를 보고 어떤 메뉴들이 있는지 기억해놔.
                2. "PDF 내용"을 전체적으로 본 후에, 각 {menu}에 알맞은 내용을 300자 정도 작성해줘.
                3. 코드는 절대 생성하면 안돼.
        """
        

        payload = {
            "model": model,  # 사용 중인 Ollama 모델 이름으로 변경하세요
            "prompt": prompt
        }
        try:
            print("start response : ", len(prompt))
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
                
            all_text = parse_response(all_text)
            all_text = all_text[0]
            print("all_text :", all_text[0])
            return all_text.strip() if all_text else "Empty response received"

        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 실패: {e}")
            raise RuntimeError(f"Ollama API 요청 실패: {e}")
        
        
def parse_response(response_text):
    """
    응답 텍스트에서 JSON 객체를 추출하여 파싱하는 함수
    Args:
        response_text (str): Ollama API로부터 받은 응답 텍스트

    Returns:
        dict: 파싱된 JSON 객체

    Raises:
        ValueError: 응답이 유효한 JSON 형식이 아닐 경우
    """
    try:
        # JSON 코드 블록 추출
        print("원본 데이터 :", response_text)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        print(json_match,"<====json_match")
        # 코드 블록이 없는 경우, 첫 번째 JSON 객체 추출 시도
        json_objects = re.findall(r'\{.*?\}', response_text, re.DOTALL)
        print(json_objects,"<====json_objects")
        for obj in json_objects:
            try:
                if "title_structure" in obj:
                    return json_objects
            except json.JSONDecodeError:
                continue  # 유효한 JSON이 아니면 건너뜀
        # 모든 시도가 실패한 경우 예외 발생
        raise ValueError("응답 내에 유효한 JSON 객체를 찾을 수 없습니다.")
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        raise ValueError("응답이 유효한 JSON 형식이 아닙니다.")

class OllamaLLM(BaseLLM):
    client: OllamaClient = Field(..., description="OllamaClient instance")
    model_name: str = Field(default="bllossom", description="Model name to use with Ollama")
    # model_name: str = Field(default="llama3.2", description="Model name to use with Ollama")

    def __init__(self, **data):
        super().__init__(**data)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"OllamaLLM._call invoked with prompt: {prompt}")
        answer = self.client.generate(model=self.model_name, prompt=prompt)
        print(f"OllamaLLM._call received answer: {answer}")
        return answer

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        answer = self._call(prompt, stop)
        return answer

    
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResult:
        responses = [self._call(prompt, stop) for prompt in prompts]
        return LLMResult(generations=[[{"text": resp}] for resp in responses])
    
    # streaming 방식
    # def generate(self, model: str, prompt: str) -> str:
    #     """
    #     Ollama API를 사용하여 텍스트 생성
    #     Args:
    #         model (str): 사용할 Ollama 모델 이름
    #         prompt (str): 입력 프롬프트

    #     Returns:
    #         str: Ollama 모델의 생성된 텍스트
    #     """
    #     payload = {
    #         "model": model,
    #         "prompt": prompt
    #     }
    #     print("check payload")
    #     try:
    #         print(self.api_url, payload, "response")
    #         with requests.post(self.api_url, json=payload, stream=True) as response:
    #             response.raise_for_status()  # HTTP 에러 발생 시 예외 처리
    #             generated_text = ""
    #             for line in response.iter_lines(decode_unicode=True):
    #                 if line.strip():  # 빈 줄 제거
    #                     try:
    #                         print("Streamed line:", line)  # 디버깅용 출력
    #                         json_line = json.loads(line)
    #                         generated_text += json_line.get("response", "")
    #                     except json.JSONDecodeError:
    #                         print(f"Invalid JSON line skipped: {line}")
    #                         continue  # JSON 파싱 오류 시 건너뛰기
    #             return generated_text  # 전체 결과 반환
    #     except requests.exceptions.RequestException as e:
    #         raise RuntimeError(f"Ollama API 요청 실패: {e}")
    
    # 문장 단위 스트리밍 처리
    # def generate(self, model: str, prompt: str) -> str:
    #     payload = {
    #         "model": model,
    #         "prompt": prompt
    #     }
    #     try:
    #         print("Payload being sent:", payload)
    #         response = requests.post(self.api_url, json=payload, stream=True)
    #         response.raise_for_status()

    #         # 스트리밍 처리
    #         generated_text = ""
    #         sentence_buffer = ""
    #         for line in response.iter_lines(decode_unicode=True):
    #             if line.strip():
    #                 try:
    #                     print("Streamed line:", line)  # 디버깅용
    #                     json_line = json.loads(line.strip())  # JSON 파싱
    #                     chunk = json_line.get("response", "")

    #                     # 문장 완성 체크
    #                     sentence_buffer += chunk
    #                     while any(punct in sentence_buffer for punct in [".", "?", "!"]):  # 마침표 등 기준으로 문장 단위 분리
    #                         for punct in [".", "?", "!"]:
    #                             if punct in sentence_buffer:
    #                                 sentence, sentence_buffer = sentence_buffer.split(punct, 1)
    #                                 sentence += punct
    #                                 print("Completed sentence:", sentence.strip())  # 문장 단위 디버깅
    #                                 generated_text += sentence.strip() + "\n"
    #                 except json.JSONDecodeError as e:
    #                     print(f"Invalid JSON line skipped: {line}, Error: {e}")
    #                     continue

    #         # 마지막 남은 문장 추가
    #         if sentence_buffer.strip():
    #             generated_text += sentence_buffer.strip()

    #         return generated_text if generated_text else "Empty response received"

    #     except Exception as e:
    #         print(f"Error: {e}")
    #         raise RuntimeError(f"Ollama API 요청 실패: {e}")
    
