# Configuration management for llm-inference-runtime

from functools import lru_cache
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model Configuration
    model: str = Field(..., description="HuggingFace model ID or local path")
    tensor_parallel_size: int = Field(default=1, ge=1, le=8, description="Number of GPUs for tensor parallelism")
    pipeline_parallel_size: int = Field(default=1, ge=1, le=8, description="Number of GPUs for pipeline parallelism")
    max_batch_size: int = Field(default=256, ge=1, le=8192, description="Max tokens per batch")
    max_seq_len: int = Field(default=4096, ge=1, le=131072, description="Max sequence length")
    max_num_seqs: int = Field(default=256, ge=1, le=4096, description="Max concurrent sequences")
    dtype: str = Field(default="auto", description="Model dtype (auto, half, bfloat16, float16, float32)")
    quantization: str = Field(default="none", description="Quantization method (none, awq, gptq, sq, marlin)")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=0.99, description="Fraction of GPU memory to use")
    swap_space: int = Field(default=4, ge=0, le=64, description="CPU swap space in GB")
    trust_remote_code: bool = Field(default=False, description="Allow remote code execution")
    disable_log_stats: bool = Field(default=False, description="Disable vLLM stats logging")
    enable_chunked_prefill: bool = Field(default=True, description="Enable chunked prefill for long contexts")
    enable_prefix_caching: bool = Field(default=True, description="Enable automatic prefix caching")
    enforce_eager: bool = Field(default=False, description="Disable CUDA graph capture")
    max_model_len: Optional[int] = Field(default=None, description="Override max_model_len from model config")

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Host to bind")
    port: int = Field(default=8000, ge=1, le=65535, description="HTTP port")
    workers: int = Field(default=1, ge=1, le=4, description="Uvicorn workers (use 1 for vLLM)")
    log_level: str = Field(default="info", description="Logging level")

    # Authentication
    api_keys: str = Field(default="", description="Comma-separated API keys (optional)")
    rate_limit_rpm: int = Field(default=1000, ge=1, description="Requests per minute per key")
    rate_limit_tpm: int = Field(default=100000, ge=1, description="Tokens per minute per key")

    # Monitoring
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=8000, description="Metrics port (same as HTTP)")

    # Redis (optional, for caching/queue)
    redis_url: str = Field(default="", description="Redis URL (optional)")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # HuggingFace
    hf_token: str = Field(default="", description="HuggingFace token for private models")
    hf_home: str = Field(default="/root/.cache/huggingface", description="HF cache directory")

    @field_validator("api_keys")
    @classmethod
    def parse_api_keys(cls, v: str) -> List[str]:
        if not v:
            return []
        return [k.strip() for k in v.split(",") if k.strip()]

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        allowed = {"auto", "half", "float16", "bfloat16", "float32", "float"}
        if v.lower() not in allowed:
            raise ValueError(f"dtype must be one of {allowed}")
        return v.lower()

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str) -> str:
        allowed = {"none", "awq", "gptq", "squeezellm", "marlin", "gguf", "bitsandbytes"}
        if v.lower() not in allowed:
            raise ValueError(f"quantization must be one of {allowed}")
        return v.lower()


@lru_cache
def get_settings() -> Settings:
    return Settings()