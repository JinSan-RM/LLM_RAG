"""This module contains configs for openai API"""
import os

class OpenAIConfig:
    """Necessary configs for OpenAI API.

    Attributes:
        openai_api_key [required] (str): OpenAI API key.
        openai_api_base [optional] (str): Base URL for the API.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY", None)
    openai_api_base = os.getenv("OPENAI_API_BASE", None)

    def __init__(
        self,
        openai_api_key: str = None,
        openai_api_base: str = None
    ) -> None:
        # 만약 인스턴스 생성 시 매개변수가 전달되면 우선 사용하고, 아니면 클래스에 설정된 기본값 사용
        self.openai_api_key = openai_api_key or self.openai_api_key
        self.openai_api_base = openai_api_base or self.openai_api_base
