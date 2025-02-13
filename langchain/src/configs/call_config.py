import os
from typing import Union, Optional, Dict
from pydantic import BaseModel, validator
from pydantic.error_wrappers import ValidationError
from bson import ObjectId


class Completions(BaseModel):
    """Custom class for Completions data"""
    model: str = "text-davinci-003"
    prompt: Union[str, list] = "Hey, How are you?"
    suffix: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = None
    stop: Optional[Union[str, list]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
    logit_bias: Optional[dict] = None
    user: Optional[str] = None


class ChatCompletions(BaseModel):
    """Custom class for Chat Completions data"""
    model: str = "gpt-3.5-turbo"
    messages: list = [{"role": "user", "content": "Hey, How are you?"}]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[Union[str, list]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict] = None
    user: Optional[str] = None


class Embeddings(BaseModel):
    """Custom class for Chat Completions data"""
    model: str = "text-embedding-ada-002"
    input: Union[str, list] = "Random text to test embedding"
    user: Optional[str] = None


class FineTunes(BaseModel):
    """Custom class for Fine Tuning"""
    training_file: str
    validation_file: Optional[str] = None
    model: Optional[str] = None
    n_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate_multiplier: Optional[float] = None
    prompt_loss_weight: Optional[float] = 0.01
    compute_classification_metrics: Optional[bool] = None
    classification_n_classes: Optional[int] = None
    classification_positive_class: Optional[str] = None
    classification_betas: Optional[list] = None
    suffix: Optional[str] = None