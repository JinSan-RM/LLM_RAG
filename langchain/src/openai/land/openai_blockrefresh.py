import asyncio
import json
import re

class OpenAIBlockRefresh:
    def __init__(self, batch_handler):
        self.batch_handler = batch_handler

    async def send_request(self, sys_prompt: str, usr_prompt: str, max_tokens: int = 100, extra_body: dict = None) -> str:
        model = "/usr/local/bin/models/gemma-3-4b-it"
        response = await asyncio.wait_for(
            self.batch_handler.process_single_request({
                "sys_prompt": sys_prompt,
                "usr_prompt": usr_prompt,
                "extra_body": extra_body,
                "max_tokens": max_tokens,
                "model": model,
                "temperature": 0.8,
                "top_p": 0.6,
                "n": 1,
                "stream": False,
                "logprobs": None
            }, request_id=0),
            timeout=30
        )
        return response

    async def block_refresh(self, section: str, content: str, block):
        try:
            # 태그 이름과 기존 콘텐츠 추출
            
            section = list(context.keys())[0]
            content = list(context.values())[0]
            length = int(list(text_length)[0])
            tag = list(tag)[0]
            # 시스템 프롬프트와 사용자 프롬프트 작성
            sys_prompt = "자연스럽고 문맥에 맞는 텍스트를 생성해주세요."
            usr_prompt = f"""
            다음 문맥과 기존 텍스트를 고려하여 자연스러운 텍스트를 생성해주세요:
            
            [섹션 : {section}]
            
            [문맥: {content}]
            
            
            [기존 텍스트 (HTML {tag} 내): {text_box}]
            
            위 정보를 바탕으로 약 {length}자 분량의 텍스트를 생성해주세요.
            
            중요 가이드라인:
            - 정확히 {length}자일 필요는 없습니다
            - {length}의 -15%~0% 범위 내에서 자연스럽게 작성하세요
            - 내용의 완결성과 文맥 유지를 우선시하세요
            - 너무 짧거나 너무 길지 않게 균형 있는 분량으로 작성하세요
            - 해당 텍스트는 HTML {tag} 내에 삽입될 예정입니다
            
            위 가이드라인에 맞춰 자연스럽고 유동적인 텍스트를 생성해주세요.
            """

            # extra_body에 태그 이름 설정
            extra_body = {
                "guided_json": {
                    "type": "object",
                    "properties": {
                        tag: {
                            "type": "string",
                        }
                    },
                    "required": [tag]
                }
            }

            # send_request 호출
            result = await self.send_request(
                sys_prompt,
                usr_prompt,
                max_tokens=200,
                extra_body=extra_body
            )
            result.data['generations'][0][0]['text'] = self.extract_json(result.data['generations'][0][0]['text'])  # JSON 추출
            # 결과는 {tag_name: generated_text} 형식으로 구성}

            return result

        except Exception as e:
            print(f"Exception occurred while regenerating: {str(e)}")
            return None
        
    def extract_json(self, text):
        # 가장 바깥쪽의 중괄호 쌍을 찾습니다.
        text = re.sub(r'[\n\r\\\\/]', '', text, flags=re.DOTALL)
        json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            # Handle case where only opening brace is found
            json_str = re.search(r'\{.*', text, re.DOTALL)
            if json_str:
                json_str = json_str.group() + '}'
            else:
                return None

        # Balance braces if necessary
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try a more lenient parsing approach
            try:
                return json.loads(json_str.replace("'", '"'))
            except json.JSONDecodeError:
                return None