"""This module handles openai requests."""
from langchain_openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from src.configs.openai_config import OpenAIConfig
import asyncio

# import openai
# import tiktoken
# from logger.ve_logger import VeLogger
# from openai import AsyncOpenAI

from typing import Optional

from pydantic import BaseModel, Field

class Pydantic_Test(BaseModel):

    question: str = Field(description="The question")
    answer: str = Field(description="The answer")

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
            # openai_api_base=openai_config.openai_api_base,
            openai_api_base="http://vllm:8002/v1",
            streaming=streaming,
            max_tokens=2000
        )

        self.chatmsg = ChatOpenAI(
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
        
        # # NOTE : 임시 테스트용용
        # self.chatgpt = ChatOpenAI(
        #     model="gpt-4o",
        #     openai_api_key="",
        #     openai_api_base="https://api.openai.com/v1"
        # )        
        
    async def completions(self, **kwargs):
        max_tokens = kwargs.get("max_tokens", self.llm.max_tokens)
        try:
            # print(f"Calling completions with kwargs: {kwargs}")
            if self.streaming:
                response = await self.stream_completion(**kwargs)
            else:
                # NOTE : 여기서 arguments 체크해보기
                print("generate 할 때 여기가 돌아가는게 맞는건가??")
                # print("kwargs : ", kwargs)
                # print("[kwargs['prompt']] : ", [kwargs['prompt']])
                
                test_data = [
                    (
                        "system",
                        """
                                    You are a professional in business plan writing. 
            You are provided with user input in the form of a sentence or paragraph. 
            Your task is to write a narrative paragraph to assist in creating a business plan based on this input. 
            Follow these instructions precisely:

            #### INSTRUCTIONS ####
            
            STEP 1. Identify and include key information from the user input, such as below. If the usr_input is not enough, you can fill it yourself.
                1) BUSINESS ITEM: Specific product or service details
                2) SLOGAN OR CATCH PHRASE: A sentence expressing the company's main vision or ideology
                3) TARGET CUSTOMERS: Characteristics and needs of the major customer base
                4) CORE VALUE PROPOSITION: Unique value provided to customers
                5) PRODUCT AND SERVICE FEATURES: Main functions and advantages
                6) BUSINESS MODEL: Processes that generate profits by providing differentiated value
                7) PROMOTION AND MARKETING STRATEGY: How to introduce products or services to customers            
            STEP 2. Develop the business plan narrative step-by-step using only the keywords and details from the user input. Do not expand the scope beyond the provided content.
            STEP 3. Write a paragraph of 1000 to 1500 characters to ensure the content is detailed and informative.
            STEP 4. Avoid repeating the same content to meet the character limit. Use varied expressions and vocabulary to enrich the narrative.
            STEP 5. Do not include repeated User-Assistant interactions or unnecessary filler. End the output naturally after completing the narrative.
            STEP 6. Ensure the text is free of typos and grammatical errors.
            STEP 7. Output only the final business plan narrative text. Do not include tags (e.g., [SYSTEM], <|eot_id|>), JSON formatting (e.g., {{Output: "..."}}), or any metadata in the output.
            STEP 8. 출력은 반드시 **한국어**로 해.
                        """
                    ),
                    ("human", "abc 우리 기업의 이름은 '제주 귤 농장'입니다. 참고해서 작업해주세요.")
                ]
                
                test_data2 = "Respond **only** with a valid JSON object containing 'question' and 'answer' keys, no additional text. Example: {\"question\": \"한국의 수도는 어디인가요?\", \"answer\": \"서울\"}"
                
                test_data3 = [
                                (
                                    "system",
                                    "You are a helpful translator. Translate the user sentence to Korean.",
                                ),
                                ("human", "can you translate this sentence?"),
                            ]
                
                test_data3_2 = [
                    SystemMessage(content="You are a helpful translator. Translate the user sentence to Korean."),
                    HumanMessage(content="너가 이전에 했던 대답 알려줘"),
                ]                
                
                # structured_llm_gpt = self.chatgpt.with_structured_output(Pydantic_Test)
                # response = structured_llm_gpt.invoke(test_data2)
                # response = await asyncio.to_thread(structured_llm_gpt.invoke, input=test_data2)
                
                
                # structured_llm = self.chat.with_structured_output(Pydantic_Test)
                # response = await structured_llm.invoke(test_data2)     
                           
                # structured_llm = self.chat.with_structured_output(Pydantic_Test)
                # response = await structured_llm.invoke(test_data3)
                # response = await asyncio.to_thread(structured_llm.invoke, input=test_data3)
                
                # response = await asyncio.to_thread(structured_llm.invoke, input=test_data)
                # response = await asyncio.to_thread(self.chat.invoke, input=test_data2)
                # response = await asyncio.to_thread(self.llm.invoke, input=test_data2)
                # response = await asyncio.to_thread(self.llm.generate, prompts=[test_data2])
                # response = await asyncio.to_thread(self.llm.generate, [kwargs['prompt']], max_tokens=max_tokens)
                
                test_data4 = "Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]"
                # response = await asyncio.to_thread(self.chatmsg.invoke, input=test_data3)
                # NOTE : 다음 메시지가 나오며 안됨. OpenAI API call failed: 'RunnableBinding' object is not callable
                # jsonmode_chat = self.chat.bind(response_format={"type": "json_object"})
                # response = await asyncio.to_thread(jsonmode_chat.invoke, input=test_data)
                
                response = await asyncio.to_thread(self.chatguided.invoke, input=test_data3_2)
                
                # structured_llm_gpt = self.chatguided.with_structured_output(Pydantic_Test)
                # response = await asyncio.to_thread(structured_llm_gpt.invoke, input=test_data3_2)
                
                print("TYPE_어떤게 반환되려나? : ", type(response))
                print("어떤게 반환되려나? : ", response)
                print("어떤게 반환되려나?_type(response.content) : ", type(response.content))
                print("어떤게 반환되려나?_response.content : ", response.content)
                
            # print(f"Completions response: {response}")
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
            print(f"Calling chat completions with kwargs: {kwargs}")
            messages = [
                SystemMessage(content=kwargs.get('system_message', "You are a helpful assistant.")),
                HumanMessage(content=kwargs['messages'][-1]['content'])
            ]
            
            if self.streaming:
                async for chunk in self.chat.astream(messages):
                    yield chunk.content
            else:
                response = await self.chat.ainvoke(messages)
                print(f"Chat completions response: {response}")
                yield response.content  # Assuming response has a 'content' attribute
        except Exception as e:
            print(f"OpenAI API call failed: {str(e)}")
            raise
                
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
