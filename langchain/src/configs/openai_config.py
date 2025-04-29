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


# TODO 250429 : 일단은 가장 말단에 req.standard_country_code를 이용해서 할 건데, 나중에는 main에서부터 받아서 분기 태우기
class OpenAIConfig_Language:
    """Necessary configs for OpenAI API.

    Attributes:
        openai_api_key [required] (str): OpenAI API key.
        openai_api_base [optional] (str): Base URL for the API.
    """
    Language_KR = "출력은 반드시 한국어로 해줘."
    Language_JP = "出力は日本語でなければなりません。"
    Language_US = "The output language must be English."

    def __init__(
        self,
        Language_KR: str = None,
        Language_JP: str = None,
        Language_US: str = None
    ) -> None:
        # 만약 인스턴스 생성 시 매개변수가 전달되면 우선 사용하고, 아니면 클래스에 설정된 기본값 사용
        self.Language_KR = Language_KR or self.Language_KR
        self.Language_JP = Language_JP or self.Language_JP
        self.Language_US = Language_US or self.Language_US
