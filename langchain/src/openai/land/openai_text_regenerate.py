import asyncio
import json

class OpenAITextRegenerator:
    
    def __init__(self, requests, batch_handler):
        self.requests = requests
        self.batch_handler = batch_handler
    
    async def send_request(self, sys_prompt: str, usr_prompt: str, max_tokens: int = 100, extra_body: dict = None) -> str:
                
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                    # "prompt": prompt,
                    "sys_prompt": sys_prompt,
                    "usr_prompt": usr_prompt,
                    "extra_body": extra_body,
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                    "top_p": 0.1,
                    "n": 1,
                    "stream": False,
                    "logprobs": None
                }, request_id=0),
                timeout=30  # 적절한 타임아웃 값 설정
            )
            return response

    
    async def regenerate(self, request_id, request):
        """Regenerate the request using the batch handler."""
        try:
            # Extract necessary information from the request
            
            # Call the batch handler to process the request
            result = await self.batch_handler.process_single_request(request, request_id)
            
            if result.success:
                # Process the successful result (e.g., save it, log it, etc.)
                print(f"Request {request_id} processed successfully: {result.data}")
            else:
                # Handle errors (e.g., log them, retry, etc.)
                print(f"Error processing request {request_id}: {result.error}")
        
        except Exception as e:
            print(f"Exception occurred while regenerating request {request_id}: {str(e)}")