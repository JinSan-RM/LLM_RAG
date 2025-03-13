"""This module handles openai requests."""
from langchain_openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from src.configs.openai_config import OpenAIConfig
import asyncio

# import openai
# import tiktoken
# from logger.ve_logger import VeLogger
# from openai import AsyncOpenAI


class OpenAIService:
    """This class handles openai requests."""

    # Initialize logger

    def __init__(
        self,
        openai_config: OpenAIConfig = None,
        streaming: bool = False
        ) -> None:
        """Intializer method of the class

        Args:
            openai_config (OpenAIConfig): Neccessary configs for openai api.

        Returns:
            None

        """
        # Check Arguments
        if openai_config is None:
            raise ValueError(
                "Provide openai_config when initializing class."
                )

        if openai_config.openai_api_key is None:
            raise ValueError(
                "Provide OpenAI API key when initializing class. You can set the " \
                "enviroment variable `OPENAI_API_KEY` to your OpeAI API key."
            )
        if openai_config.openai_api_base is None:
            raise ValueError(
                "Not Enter src/openai_config.py -> base url not found"
                )
        self.streaming = streaming
        callback_manager = AsyncCallbackManager([StreamingStdOutCallbackHandler()]) if streaming else CallbackManager([])
        
        self.llm = OpenAI(
            model="/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
            openai_api_key=openai_config.openai_api_key,
            openai_api_base=openai_config.openai_api_base,
            streaming=streaming,
            callback_manager=callback_manager,
            max_tokens=2000
        )
        
        self.chat = ChatOpenAI(
            model="/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
            openai_api_key=openai_config.openai_api_key,
            openai_api_base=openai_config.openai_api_base,
            streaming=streaming,
            max_tokens=2000
        )
        self.chatguided = ChatOpenAI(
            model="/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
            openai_api_key=openai_config.openai_api_key,
            openai_api_base="http://vllm:8002/v1",
            extra_body={"guided_json": {
                "type": "object",
                "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
                "required": ["question", "answer"]
            }}
        )
        
    async def completions(self, **kwargs):
        max_tokens = kwargs.get("max_tokens", self.llm.max_tokens)
        try:
            # print(f"Calling completions with kwargs: {kwargs}")
            if self.streaming:
                response = await self.stream_completion(**kwargs)
            else:
                response = await asyncio.to_thread(self.llm.generate, [kwargs['prompt']], max_tokens=max_tokens)
            # print(f"Completions response: {response}")
            return response
        except Exception as e:
            print(f"OpenAI API call failed: {str(e)}")
            raise
        
    async def text_invoke(self, **kwargs):
        max_tokens = kwargs.get("max_tokens", self.chatguided.max_tokens)
        try:
            print("invoke 타이밍")
            if self.streaming:
                response = await self.stream_chat_completion(**kwargs)
            else:
                # 'messages' 키를 사용하고, 리스트로 전달
                messages = kwargs.get('messages', [])
                if not messages:
                    raise ValueError("No 'messages' provided in the request")
                # chat.invoke가 리스트를 기대한다고 가정
                response = await asyncio.to_thread(self.chatguided.invoke, messages, max_tokens=max_tokens)
            return response
        except Exception as e:
            print(f"OpenAI Chat API call failed : {str(e)}")
            raise

    async def stream_completion(self, **kwargs):
        response = ""
        async for chunk in self.llm.astream(kwargs['prompt']):
            response += chunk
            print(chunk, end="", flush=True)
        return response
    
    async def chat_completions(self, **kwargs):
        try:
            print(f"Calling chat completions with kwargs: {kwargs}")
            
            # kwargs에서 sys_prompt와 usr_prompt 추출, 기본값 설정
            sys_prompt = kwargs.get('sys_prompt')
            usr_prompt = kwargs.get('usr_prompt')
            # extra_body = kwargs.get('extra_body')
            max_tokens = kwargs.get('max_tokens')
            
            # usr_prompt가 없으면 messages에서 마지막 content를 사용 (기존 로직 유지)
            if not usr_prompt and 'messages' in kwargs and kwargs['messages']:
                usr_prompt = kwargs['messages'][-1]['content']
            
            if not usr_prompt:
                raise ValueError("No user prompt provided in 'usr_prompt' or 'messages'")

            # messages 리스트 구성
            messages = [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=usr_prompt)
            ]
            
            # 비스트리밍 방식으로 응답 생성
            if 'extra_body' in kwargs:
                if extra_body and "guided_json" in extra_body:
                    sys_prompt += f"\nRespond in JSON format with the 'generate' field containing a narrative paragraph of {max_tokens} characters in Korean, following the instructions below."
                extra_body = kwargs.get('extra_body')
                response = await self.chat.ainvoke(
                    input=messages,
                    max_tokens=max_tokens,
                    extra_body=extra_body
                )
            else:
                response = await self.chat.ainvoke(
                    input=messages,
                    max_tokens=max_tokens
                )
            print(f"Chat completions response: {response}")
            return response  # response.content를 직접 반환
        
        except Exception as e:
            print(f"OpenAI API call failed: {str(e)}")
            raise
    # NOTE : 기존의 chat_completions 방식 아래래
    # async def chat_completions(self, **kwargs):
    #     try:
    #         print(f"Calling chat completions with kwargs: {kwargs}")
    #         messages = [
    #             SystemMessage(content=kwargs.get('system_message', "You are a helpful assistant.")),
    #             HumanMessage(content=kwargs['messages'][-1]['content'])
    #         ]
            
    #         if self.streaming:
    #             async for chunk in self.chat.astream(messages):
    #                 yield chunk.content
    #         else:
    #             response = await self.chat.ainvoke(messages)
    #             print(f"Chat completions response: {response}")
    #             yield response.content  # Assuming response has a 'content' attribute
    #     except Exception as e:
    #         print(f"OpenAI API call failed: {str(e)}")
    #         raise
                
    async def stream_chat_completion(self, **kwargs):
        response = ""
        async for chunk in await self.chat.chat.completions.create(**kwargs, stream=True):
            content = chunk.choices[0].delta.content
            if content:
                response += content
                print(content, end="", flush=True)
        return response
    # =====================================================================
    #     # self.client = OpenAI(
    #     #     api_key=openai_config.openai_api_key,
    #     #     base_url=openai_config.openai_api_base
    #     # )
    #     self.client = AsyncOpenAI(
    #         api_key=openai_config.openai_api_key,
    #         base_url=openai_config.openai_api_base
    #     )

    # async def completions(self, *args, **kwargs):
    #     try:
    #         print(f"Calling completions with args: {args}, kwargs: {kwargs}")
    #         response = await self.client.completions.create(**kwargs)
    #         print(f"Completions response: {response}")
    #         return response
    #     except Exception as e:
    #         print(f"OpenAI API call failed: {str(e)}")
    #         raise

    # 이전 진행 방식.
    # =====================================================================
    # def _validate_model(self, model: str):
    #     """Validate model name"""
    #     if model in self.model_list:
    #         return True
    #     return False

    # def _count_tokens(self, text: str, model: str):
    #     """Count tokens."""
    #     if self._validate_model(model):
    #         tokenizer = tiktoken.encoding_for_model(model)
    #         return len(tokenizer.encode(text))

    # def embeddings(self, *args, **kwargs):
    #     """Embedding Completion models method"""
    #     return openai.Embedding.create(*args, **kwargs)

    # def upload_files(self, *args, **kwargs):
    #     """Upload file.

    #     Args:
    #         file (any): File to upload.
    #         purpose (str): Purpose for file (Default: 'fine-tune').

    #     Returns:
    #         dict: Response from OpenAI

    #     """
    #     return openai.File.create(*args, **kwargs)

    # def list_files(self):
    #     """Get the list of uploaded files.

    #     Args:
    #         None

    #     Returns:
    #         dict: Response from OpenAI

    #     """
    #     return openai.File.list()

    # def fine_tunes(self, *args, **kwargs):
    #     """Finetune model via uploaded file.

    #     Args:
    #         None

    #     Returns:
    #         dict: Response from OpenAI

    #     """
    #     return openai.FineTune.create(*args, **kwargs)

    # def retrieve_fine_tune(self, id: str):
    #     """Get the details of a finetuned model.

    #     Args:
    #         id (str): id of the fine-tuning process.

    #     Returns:
    #         dict: Response from OpenAI

    #     """
    #     return openai.FineTune.retrieve(id=id)

    # def cancel_fine_tune(self, id: str):
    #     """Cancel a finetune process.

    #     Args:
    #         id (str): id of the fine-tuning process.

    #     Returns:
    #         dict: Response from OpenAI

    #     """
    #     return openai.FineTune.cancel(id=id)

    # def list_fine_tunes(self):
    #     """Get the list of finetuned models.

    #     Args:
    #         None

    #     Returns:
    #         dict: Response from OpenAI

    #     """
    #     return openai.FineTune.list()
