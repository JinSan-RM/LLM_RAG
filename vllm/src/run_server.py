from vllm.entrypoints.openai.api_server import run_server
from vllm import SamplingParams
import sys
import os

import argparse
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.vllm_config import VLLMConfig
# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["NCCL_DEBUG"] = "TRACE"
# os.environ["VLLM_TRACE_FUNCTION"] = "1"


def main():
    config = VLLMConfig()  # 설정 로드
    
    print("vllm container main test1111111111111111")  # 디버깅 출력
    
    # sampling_params = SamplingParams(
    #     max_tokens=2000
    # )
    # NOTE : serve는 CLI로 실행 시 사용되는 명령어
    # ※ https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    # serve(
    #     model="/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",  # Dockerfile에서 복사된 경로
    #     host=config.host,
    #     port=config.port,
    #     max_num_seqs=config.max_num_seqs,
    #     max_batch_size=config.max_batch_size,
    #     max_num_batched_tokens=config.max_num_batched_tokens,
    #     gpu_memory_utilization=1.0,
    #     request_timeout=config.request_timeout,
    #     max_waiting_tokens=config.max_waiting_tokens,
    #     sampling_params=sampling_params,
    #     quantization=config.quantization,
    #     dtype="float16",
    #     max_model_len=4096
    # )
    
    # parser = argparse.ArgumentParser(description="vLLM OpenAI API Server")
    # parser.add_argument("--model", default=config.model, type=str)
    # parser.add_argument("--host", default=config.host, type=str)
    # parser.add_argument("--port", default=config.port, type=int)
    # parser.add_argument("--max-num-seqs", default=config.max_num_seqs, type=int)
    # parser.add_argument("--max-batch-size", default=config.max_batch_size, type=int)
    # parser.add_argument("--max-num-batched-tokens", default=config.max_num_batched_tokens, type=int)
    # parser.add_argument("--gpu-memory-utilization", default=1.0, type=float)
    # parser.add_argument("--request-timeout", default=config.request_timeout, type=int)
    # parser.add_argument("--max-waiting-tokens", default=config.max_waiting_tokens, type=int)
    # parser.add_argument("--quantization", default=config.quantization, type=str)
    # parser.add_argument("--dtype", default="float16", type=str)
    # parser.add_argument("--max-model-len", default=4096, type=int)
    
    # args = parser.parse_args()
    # asyncio.run(run_server(args))
    
    
if __name__ == "__main__":
    print("vllm container main test222222222222")
    main()
    print("vllm container main test3333333333333")