"""This module handles openai requests."""
from openai import OpenAI
from openai import AsyncOpenAI
from src.configs.openai_config import OpenAIConfig
# import openai
# import tiktoken
# from logger.ve_logger import VeLogger


class OpenAIService:
    """This class handles openai requests."""

    # Initialize logger

    def __init__(self, openai_config: OpenAIConfig = None) -> None:
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

        # self.client = OpenAI(
        #     api_key=openai_config.openai_api_key,
        #     base_url=openai_config.openai_api_base
        # )
        self.client = AsyncOpenAI(
            api_key=openai_config.openai_api_key,
            base_url=openai_config.openai_api_base
        )

    async def completions(self, *args, **kwargs):
        try:
            print(f"Calling completions with args: {args}, kwargs: {kwargs}")
            response = await self.client.completions.create(**kwargs)  # *args 제거
            print(f"Completions response: {response}")
            return response
        except Exception as e:
            print(f"OpenAI API call failed: {str(e)}")
            raise

    async def chat_completions(self, *args, **kwargs):
        """Chat Completion models method"""
        return await self.client.chat.completions.create(*args, **kwargs)

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
