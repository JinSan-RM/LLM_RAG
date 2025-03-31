from typing import Union, Optional, Dict, Any
from pydantic import BaseModel, validator, Field


class Completions(BaseModel):
    """Custom class for Completions data"""
    model: str = "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0"
    prompt: Union[str, list] = "Hey, How are you?"
    suffix: Optional[str] = None
    max_tokens: int = 2048  # 기본 토큰 수 (예: 16)
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, list]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    logit_bias: Optional[dict] = None
    user: Optional[str] = None
    pdf_data1: Optional[str] = None
    pdf_data2: Optional[str] = None
    pdf_data3: Optional[str] = None
    usr_msg: Optional[str] = None
    block: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    # NOTE : F.E에서 각각 텍스트 블록의 text length를 정해서 주므로 태그와 매칭해서 보내주기로 하였음. 그러면 select_block 필요 x.
    #        tag_length 새로 생성
    select_block: Optional[Dict[str, str]] = Field(default_factory=dict)
    tag_length: Optional[Dict[str, Any]] = Field(default_factory=dict)
    section_context: Optional[Dict[str, str]] = Field(default_factory=dict)
    all_usr_data: Optional[str] = None
    section_html: Optional[str] = None

class ChatCompletions(BaseModel):
    """Custom class for Chat Completions data"""
    model: str = "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0"
    messages: list = [{"role": "user", "content": "Hey, How are you?"}]
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, list]] = None
    max_tokens: int = 2048  # Chat의 경우, 좀 더 많은 토큰을 생성할 수 있도록 기본값 설정
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[dict] = None
    user: Optional[str] = None


# class Embeddings(BaseModel):
#     """Custom class for Chat Completions data"""
#     model: str = "text-embedding-ada-002"
#     input: Union[str, list] = "Random text to test embedding"
#     user: Optional[str] = None


# class FineTunes(BaseModel):
#     """Custom class for Fine Tuning"""
#     training_file: str
#     validation_file: Optional[str] = None
#     model: Optional[str] = None
#     n_epochs: Optional[int] = None
#     batch_size: Optional[int] = None
#     learning_rate_multiplier: Optional[float] = None
#     prompt_loss_weight: Optional[float] = 0.01
#     compute_classification_metrics: Optional[bool] = None
#     classification_n_classes: Optional[int] = None
#     classification_positive_class: Optional[str] = None
#     classification_betas: Optional[list] = None
#     suffix: Optional[str] = None