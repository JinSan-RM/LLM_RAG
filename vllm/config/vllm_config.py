from dataclasses import dataclass, field


class SamplingParams:
    max_tokens: int = 1024         # 생성할 최대 토큰 수
    temperature: float = 0.7        # 온도 (랜덤성 조절)
    top_p: float = 1.0              # nucleus sampling 확률 임계값
    top_k: int = -1                 # top-k sampling (음수면 사용하지 않음)
    # 추가적으로 필요한 파라미터들을 여기에 정의할 수 있습니다.

@dataclass
class VLLMConfig:
    # 모델 및 서버 기본 설정
    model: str = "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0"
    host: str = "0.0.0.0"
    port: int = 8002

    # 병렬 처리 관련 설정
    max_num_seqs: int = 256         # 동시 처리 가능한 최대 시퀀스 수
    max_batch_size: int = 50        # 단일 배치당 최대 요청 수
    max_num_batched_tokens: int = 2048  # 배치당 최대 토큰 수

    # 메모리 관련 설정
    gpu_memory_utilization: float = 1.0  # GPU 메모리 사용률
    # quantization: str = "awq"            # 양자화 방식

    # 스케줄링 관련 설정
    request_timeout: int = 120      # 요청 타임아웃(초)
    max_waiting_tokens: int = 20    # KV cache 대기 토큰 수

    sampling_params: SamplingParams = field(default_factory=SamplingParams)
