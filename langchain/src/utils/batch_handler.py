# src/utils/batch_handler.py
from dataclasses import dataclass
import asyncio
import time
from typing import Dict, Any
from asyncio import TimeoutError
from src.openai.openai_api_call import OpenAIService
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class RequestResult:
    success: bool
    data: Any = None
    error: str = None
    error_details: Dict = None

class BatchRequestHandler:
    def __init__(self, openai_service: OpenAIService, 
                 max_concurrent_requests: int = 50,
                 request_timeout: int = 240,
                 requests_per_second: float = 20):  # 초당 요청 수 제한
        self.openai_service = openai_service
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_timeout = request_timeout
        self.requests_per_second = requests_per_second
        self.last_request_time = None  # 마지막 요청 시간

    async def process_single_request(self, request: Dict[str, Any],
                                   request_id: int) -> RequestResult:
        """단일 요청을 처리하는 메서드
        
        Args:
            request: 요청 데이터 (max_tokens 포함 가능)
            request_id: 요청 ID
        
        Returns:
            RequestResult: 처리 결과
        """
        try:
            # logger.debug(f"Processing request {request_id}: {request}")
            max_tokens = request.get("max_tokens", "default")  # 디버깅용으로 max_tokens 확인
            logger.debug(f"Request {request_id} using max_tokens: {max_tokens}")

            async with self.semaphore:
                # Rate limiting - wait if needed
                now = time.time()
                if self.last_request_time is not None:
                    time_since_last = now - self.last_request_time
                    min_interval = 1.0 / self.requests_per_second  # 요청 간 최소 간격
                    if time_since_last < min_interval:
                        await asyncio.sleep(min_interval - time_since_last)

                self.last_request_time = time.time()
                
                # Execute request with timeout
                try:
                    print(f"request : {request}")
                    if 'sys_prompt' in request or 'usr_prompt' in request:
                        response = await self.openai_service.chat_completions(**request)  # 단일 결과 반환
                        return RequestResult(
                            success=True,
                            data={'generations': [[{'text': response.content}]]}
                        )
                    elif 'messages' in request:
                        response = await self.openai_service.chat_completions(**request)  # 단일 결과 반환
                        return RequestResult(
                            success=True,
                            data={'choices': [{'message': {'content': response}}]}
                        )
                    else:
                        response = await asyncio.wait_for(
                            self.openai_service.completions(**request),
                            timeout=self.request_timeout
                        )
                        return RequestResult(success=True, data=response)

                except TimeoutError:
                    return RequestResult(
                        success=False,
                        error=f"Request timed out after {self.request_timeout}s",
                        error_details={"type": "timeout"}
                    )
                except Exception as e:
                    logger.error(f"Request {request_id} failed with error: {str(e)}")
                    return RequestResult(
                        success=False,
                        error=str(e),
                        error_details={
                            "type": type(e).__name__,
                            "args": getattr(e, 'args', None)
                        }
                    )
        except Exception as e:
            logger.error(f"Unexpected error in request {request_id}: {str(e)}")
            return RequestResult(
                success=False,
                error=str(e),
                error_details={
                    "type": "unexpected_error",
                    "error_type": type(e).__name__,
                    "args": getattr(e, 'args', None)
                }
            )

    async def process_batch(self, requests: list) -> dict:
        if not requests:
            return {
                "error": "No requests provided",
                "status_code": 400
            }
        
        logger.info(f"Processing batch of {len(requests)} requests")
        
        # Process all requests
        tasks = [
            self.process_single_request(req, idx)
            for idx, req in enumerate(requests)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Prepare detailed response
        response = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_requests": len(requests),
            "successful_requests": sum(1 for r in results if getattr(r, 'success', False)),
            "failed_requests": sum(1 for r in results if not getattr(r, 'success', False)),
            "results": [
                {
                    "request_id": idx,
                    "success": getattr(result, 'success', False),
                    "data": result.data if getattr(result, 'success', False) else None,
                    "error": result.error if hasattr(result, 'error') else str(result),
                    "error_details": result.error_details if hasattr(result, 'error_details') else None
                }
                for idx, result in enumerate(results)
            ]
        }
        
        # 디버깅을 위한 상세 로그
        logger.info(f"Batch processing complete. Success: {response['successful_requests']}, \n {response['results']}"
                   f"Failed: {response['failed_requests']}")
        
        if response["successful_requests"] == 0:
            error_summary = "\n".join([
                f"Request {r['request_id']}: {r['error']}"
                for r in response["results"]
                if not r['success']
            ])
            logger.error(f"All requests failed. Errors:\n{error_summary}")
            
        return response