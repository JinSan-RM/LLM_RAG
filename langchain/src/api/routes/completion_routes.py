# # src/api/routes/completion_routes.py
# from fastapi import APIRouter, HTTPException
# from src.configs.call_config import Completions
# from src.configs.openai_config import OpenAIConfig
# from src.openai.openai_api_call import OpenAIService
# from src.utils.batch_handler import BatchRequestHandler
# from typing import List
# from datetime import datetime

# router = APIRouter()

# # Initialize services
# openai_config = OpenAIConfig()
# openai_service = OpenAIService(openai_config)
# batch_handler = BatchRequestHandler(openai_service)

# @router.post("/batch_completions")
# async def batch_completions(requests: List[Completions]):
#     try:
#         # Convert Pydantic models to dictionaries
#         request_dicts = [req.dict() for req in requests]
        
#         # Process batch requests
#         response = await batch_handler.process_batch(request_dicts)
        
#         # Check if all requests failed
#         if response.get("successful_requests", 0) == 0:
#             raise HTTPException(
#                 status_code=500,
#                 detail="All batch requests failed"
#             )
            
#         return response
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )