"""This module contains configs for openai API"""
import os


class OpenAIConfig:
    """Necessary configs for OpenAI API.

    Attributes:
        openai_api_key [required] (str): OpenAI API key.

    """
    openai_api_key = str(os.getenv("OPENAI_API_KEY")) \
                    if os.getenv("OPENAI_API_KEY") else None
    
    openai_api_base = str(os.getenv('OPENAI_API_BASE')) \
                    if os.getenv("OPENAI_API_BASE") else None

    def __init__(
        self,
        openai_api_key: str = None,
        openai_api_base: str = None
    ) -> None:
        if openai_api_key:
            self.openai_api_key = openai_api_key
            self.openai_base_url = openai_api_base
