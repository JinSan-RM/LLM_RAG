from vllm.entrypoints.openai.api_server import serve
from vllm import SamplingParams
import sys
import os
from config.vllm_config import VLLMConfig
# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["NCCL_DEBUG"] = "TRACE"
# os.environ["VLLM_TRACE_FUNCTION"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    config = VLLMConfig()
    sampling_params=SamplingParams(
            # max_tokens=2000
        )
    serve(
        model=config.model,
        host=config.host,
        port=config.port,
        max_num_seqs=config.max_num_seqs,
        max_batch_size=config.max_batch_size,
        max_num_batched_tokens=config.max_num_batched_tokens,
        gpu_memory_utilization=1.0,
        request_timeout=config.request_timeout,
        max_waiting_tokens=config.max_waiting_tokens,
        sampling_params=sampling_params,
        quantization=config.quantization
    )


if __name__ == "__main__":
    main()