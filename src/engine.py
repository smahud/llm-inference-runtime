# vLLM Engine Wrapper for llm-inference-runtime

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    ModelCard,
    ModelList,
    ModelPermission,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.engine.arg_utils import AsyncEngineArgs as VLLMEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.transformers_utils.tokenizer import get_tokenizer

from .config import get_settings
from .metrics import get_metrics

logger = logging.getLogger(__name__)


class LLMEngine:
    """Wrapper around vLLM AsyncLLMEngine with lifecycle management."""

    def __init__(self):
        self.settings = get_settings()
        self.engine: Optional[AsyncLLMEngine] = None
        self.serving_chat: Optional[OpenAIServingChat] = None
        self.serving_completion: Optional[OpenAIServingCompletion] = None
        self.serving_embedding: Optional[OpenAIServingEmbedding] = None
        self.serving_models: Optional[OpenAIServingModels] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the vLLM engine and serving components."""
        if self._initialized:
            return

        logger.info(f"Initializing vLLM engine for model: {self.settings.model}")

        # Build engine args
        engine_args = VLLMEngineArgs(
            model=self.settings.model,
            tensor_parallel_size=self.settings.tensor_parallel_size,
            pipeline_parallel_size=self.settings.pipeline_parallel_size,
            dtype=self.settings.dtype if self.settings.dtype != "auto" else "auto",
            quantization=self.settings.quantization if self.settings.quantization != "none" else None,
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
            swap_space=self.settings.swap_space,
            max_model_len=self.settings.max_model_len or self.settings.max_seq_len,
            max_num_seqs=self.settings.max_num_seqs,
            max_num_batched_tokens=self.settings.max_batch_size,
            trust_remote_code=self.settings.trust_remote_code,
            disable_log_stats=self.settings.disable_log_stats,
            enforce_eager=self.settings.enforce_eager,
            enable_chunked_prefill=self.settings.enable_chunked_prefill,
            enable_prefix_caching=self.settings.enable_prefix_caching,
            tokenizer_mode="auto",
        )

        # Create engine
        self.engine = AsyncLLMEngine.from_engine_args(
            engine_args,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )

        # Get model config
        model_config = await self.engine.get_model_config()
        tokenizer = await self.engine.get_tokenizer()

        # Initialize serving components
        model_path = BaseModelPath(
            name=self.settings.model,
            model=self.settings.model,
        )

        self.serving_models = OpenAIServingModels(
            engine_client=self.engine,
            model_config=model_config,
            base_model_paths=[model_path],
            lora_modules=None,
            prompt_adapters=None,
        )

        self.serving_chat = OpenAIServingChat(
            engine_client=self.engine,
            model_config=model_config,
            models=self.serving_models,
            response_role="assistant",
            chat_template=None,
            chat_template_content_format="auto",
        )

        self.serving_completion = OpenAIServingCompletion(
            engine_client=self.engine,
            model_config=model_config,
            models=self.serving_models,
        )

        self.serving_embedding = OpenAIServingEmbedding(
            engine_client=self.engine,
            model_config=model_config,
            models=self.serving_models,
        )

        self._initialized = True
        logger.info("vLLM engine initialized successfully")

    async def shutdown(self) -> None:
        """Gracefully shutdown the engine."""
        if self.engine:
            logger.info("Shutting down vLLM engine...")
            # vLLM engine doesn't have explicit shutdown, just dereference
            self.engine = None
            self._initialized = False
            logger.info("vLLM engine shutdown complete")

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionStreamResponse, None]]:
        """Handle chat completion request."""
        if not self._initialized:
            await self.initialize()

        metrics = get_metrics()
        start_time = time.time()

        try:
            result = await self.serving_chat.create_chat_completion(request)
            duration = time.time() - start_time

            # Record metrics
            if metrics:
                metrics.record_request(
                    model=self.settings.model,
                    endpoint="chat",
                    status="success",
                    duration=duration,
                    prompt_tokens=getattr(result, "usage", {}).get("prompt_tokens", 0) if hasattr(result, "usage") else 0,
                    completion_tokens=getattr(result, "usage", {}).get("completion_tokens", 0) if hasattr(result, "usage") else 0,
                )

            return result

        except Exception as e:
            duration = time.time() - start_time
            if metrics:
                metrics.record_request(
                    model=self.settings.model,
                    endpoint="chat",
                    status="error",
                    duration=duration,
                )
            logger.error(f"Chat completion error: {e}")
            raise

    async def completion(
        self,
        request: CompletionRequest,
    ) -> Union[CompletionResponse, AsyncGenerator[CompletionStreamResponse, None]]:
        """Handle completion request."""
        if not self._initialized:
            await self.initialize()

        metrics = get_metrics()
        start_time = time.time()

        try:
            result = await self.serving_completion.create_completion(request)
            duration = time.time() - start_time

            if metrics:
                metrics.record_request(
                    model=self.settings.model,
                    endpoint="completion",
                    status="success",
                    duration=duration,
                    prompt_tokens=getattr(result, "usage", {}).get("prompt_tokens", 0) if hasattr(result, "usage") else 0,
                    completion_tokens=getattr(result, "usage", {}).get("completion_tokens", 0) if hasattr(result, "usage") else 0,
                )

            return result

        except Exception as e:
            duration = time.time() - start_time
            if metrics:
                metrics.record_request(
                    model=self.settings.model,
                    endpoint="completion",
                    status="error",
                    duration=duration,
                )
            logger.error(f"Completion error: {e}")
            raise

    async def embeddings(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Handle embeddings request."""
        if not self._initialized:
            await self.initialize()

        metrics = get_metrics()
        start_time = time.time()

        try:
            result = await self.serving_embedding.create_embedding(request)
            duration = time.time() - start_time

            if metrics:
                metrics.record_request(
                    model=self.settings.model,
                    endpoint="embeddings",
                    status="success",
                    duration=duration,
                    prompt_tokens=getattr(result, "usage", {}).get("prompt_tokens", 0) if hasattr(result, "usage") else 0,
                    completion_tokens=0,
                )

            return result

        except Exception as e:
            duration = time.time() - start_time
            if metrics:
                metrics.record_request(
                    model=self.settings.model,
                    endpoint="embeddings",
                    status="error",
                    duration=duration,
                )
            logger.error(f"Embeddings error: {e}")
            raise

    async def list_models(self) -> ModelList:
        """List available models."""
        if not self._initialized:
            await self.initialize()
        return await self.serving_models.show_available_models()

    async def get_model_info(self, model_name: str) -> ModelCard:
        """Get model information."""
        if not self._initialized:
            await self.initialize()
        models = await self.serving_models.show_available_models()
        for model in models.data:
            if model.id == model_name:
                return model
        raise ValueError(f"Model {model_name} not found")

    @property
    def is_ready(self) -> bool:
        """Check if engine is ready to serve requests."""
        return self._initialized and self.engine is not None


# Global engine instance
_engine: Optional[LLMEngine] = None


def get_engine() -> LLMEngine:
    """Get or create the global engine instance."""
    global _engine
    if _engine is None:
        _engine = LLMEngine()
    return _engine


async def init_engine() -> LLMEngine:
    """Initialize the global engine."""
    engine = get_engine()
    await engine.initialize()
    return engine


async def shutdown_engine() -> None:
    """Shutdown the global engine."""
    global _engine
    if _engine:
        await _engine.shutdown()
        _engine = None