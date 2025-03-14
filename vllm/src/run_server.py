from vllm.entrypoints.openai.api_server import run_server
from vllm import SamplingParams
import sys
import os

import argparse
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.vllm_config import VLLMConfig


def main():
    config = VLLMConfig()  # 설정 로드
    
 
if __name__ == "__main__":
    main()