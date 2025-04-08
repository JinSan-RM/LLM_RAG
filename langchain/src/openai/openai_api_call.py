"""This module handles openai requests."""
# from langchain_openai import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from src.configs.openai_config import OpenAIConfig
import asyncio
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
        
        # NOTE : 여기에서 분기를 태울까? 괜찮아보이는데?
        
        print("openai_config.openai_api_base : ", openai_config.openai_api_base)
        
        
        self.chat_EEVE = ChatOpenAI(
            model="/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
            openai_api_key=openai_config.openai_api_key,
            # openai_api_base=openai_config.openai_api_base,
            openai_api_base="http://vllm_eeve:8002/v1",
            streaming=streaming,
            max_tokens=2000
        )

        self.chat_gemma_3_4b = ChatOpenAI(
            model="/usr/local/bin/models/gemma-3-4b-it",
            openai_api_key=openai_config.openai_api_key,
            openai_api_base="http://vllm_gemma:8022/v1",
            streaming=streaming,
            max_tokens=500
        )
        

    async def completions(self, **kwargs):
        max_tokens = kwargs.get("max_tokens", self.llm.max_tokens)
        try:
            if self.streaming:
                response = await self.stream_completion(**kwargs)
            else:
                response = await asyncio.to_thread(self.llm.generate, [kwargs['prompt']], max_tokens=max_tokens)
                print(f"Completions response: {response}")
            return response
        
        except Exception as e:
            print(f"OpenAI API call failed: {str(e)}")
            raise

    async def stream_completion(self, **kwargs):
        response = ""
        async for chunk in self.llm.astream(kwargs['prompt']):
            response += chunk
            print(chunk, end="", flush=True)
        return response

    async def chat_completions(self, **kwargs):
        try:
            # 필요한 값들만 효율적으로 추출
            sys_prompt = kwargs.get('sys_prompt', '')
            usr_prompt = kwargs.get('usr_prompt', None)
            max_tokens = kwargs.get('max_tokens')
            temperature = kwargs.get('temperature')
            model = kwargs.get('model', "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0")
            # extra_body를 명시적으로 초기화
            extra_body = kwargs.get('extra_body')
            # =====================================
            # 추가적인 파라미터들
            repetition_penalty = kwargs.get('repetition_penalty')
            top_p = kwargs.get('top_p')
            n = kwargs.get('n')
            
            # usr_prompt 처리 최적화
            if not usr_prompt and 'messages' in kwargs and kwargs['messages']:
                usr_prompt = kwargs['messages'][-1]['content']
                
            if not usr_prompt:
                raise ValueError("No user prompt provided in 'usr_prompt' or 'messages'")
                
            # 메시지 생성을 최적화
            messages = [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=usr_prompt)
            ]
            
            # 단일 ainvoke 호출로 최적화
            invoke_params = {"input": messages}
            if max_tokens:
                invoke_params["max_tokens"] = max_tokens
            if extra_body:
                invoke_params["extra_body"] = extra_body
            if temperature:
                invoke_params["temperature"] = temperature
            if model:
                invoke_params["model"] = model
            
            # 추가적인 파라미터들 처리
            if repetition_penalty:
                invoke_params["repetition_penalty"] = repetition_penalty
            if top_p:
                invoke_params["top_p"] = top_p
            if n:
                invoke_params["n"] = n
            
            print(f"**invoke_params : {invoke_params}**")
            # 단일 비동기 호출
            if model == "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0":
                result = await self.chat_EEVE.ainvoke(**invoke_params)
            elif model == "/usr/local/bin/models/gemma-3-4b-it":
                result = await self.chat_gemma_3_4b.ainvoke(**invoke_params)
            else:
                result = await self.chat_EEVE.ainvoke(**invoke_params)
            return result
            
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
