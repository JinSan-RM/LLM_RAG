import asyncio
import json
import re
class OpenAITextRegenerator:
    
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

    # NOTE 250428 : request를 받고 거기서 풀어내기
    async def regenerate(self, text_box: str, section_context: dict, tag_length: dict):
        try:
            # 태그 이름과 기존 콘텐츠 추출
            
            section = list(section_context.keys())[0]
            context = list(section_context.values())[0]
            tag = list(tag_length.keys())[0]
            length = int(list(tag_length.values())[0])
        
            # NOTE 250429 : 좀 더 자유도 부분에서 체크가 필요. hyperparameter 수정 필요
            sys_prompt = f"""
            You are an expert at making sentences flow smoothly.
            Please create natural and contextual text.

            Based on the information above, create a text of approximately {length} characters.

            Important guidelines:

            - It does not have to be exactly {length} characters.
            - Write naturally within the range of -15% to 0% of {length}.
            - Prioritize completeness and context of content.
            - Write in a balanced amount, not too short or too long.
            - The text will be inserted within an HTML {tag}. But don't need to add tags like <p>.
            - 출력은 반드시 extra_body 형식을 맞추고, 언어는 **한국어**로 해줘.
            
            Please create natural and fluid text that follows the guidelines above.
            """
            
            usr_prompt = f"""
            Please generate natural text considering the following context and existing text.
            
            Section = {section}
            
            Context = {context}
            
            previous text = "HTML {tag}: {text_box}"
            
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
            
            # print("Test_result in Class : ", result)
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