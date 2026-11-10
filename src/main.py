#!/usr/bin/env python3
"""
llm-inference-runtime — High-throughput LLM Inference Gateway
Production-grade FastAPI server with vLLM engine, dynamic batching, and OpenAI-compatible API.
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.engine import init_engine, shutdown_engine, get_engine
from src.metrics import init_metrics
from src.middleware import AuthMiddleware, RateLimitMiddleware, LoggingMiddleware, ActiveRequestMiddleware
from src.health import router as health_router
from src.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelList,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting llm-inference-runtime...")
    start_time = time.time()

    # Initialize metrics
    init_metrics()
    logger.info("Metrics initialized")

    # Initialize vLLM engine
    try:
        await init_engine()
        load_time = time.time() - start_time
        logger.info(f"Engine initialized in {load_time:.2f}s")

        # Record model load time
        from src.metrics import get_metrics
        metrics = get_metrics()
        if metrics:
            metrics.record_model_load(settings.model, load_time)

    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        raise

    logger.info(f"Server ready on {settings.host}:{settings.port}")
    yield

    # Shutdown
    logger.info("Shutting down...")
    await shutdown_engine()
    logger.info("Shutdown complete")


app = FastAPI(
    title="llm-inference-runtime",
    description="High-throughput LLM Inference Gateway with vLLM",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware (order matters - auth first, then rate limit, then logging)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(ActiveRequestMiddleware)
app.add_middleware(LoggingMiddleware)

# Health routes
app.include_router(health_router)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"type": "internal_error", "message": "Internal server error"}},
    )


# Chat Completions
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    engine = get_engine()
    if not engine.is_ready:
        return JSONResponse(
            status_code=503,
            content={"error": {"type": "service_unavailable", "message": "Model not ready"}},
        )

    result = await engine.chat_completion(request)

    # Handle streaming
    if request.stream:
        from fastapi.responses import StreamingResponse

        async def stream_generator():
            async for chunk in result:
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return result


# Completions (Legacy)
@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint."""
    engine = get_engine()
    if not engine.is_ready:
        return JSONResponse(
            status_code=503,
            content={"error": {"type": "service_unavailable", "message": "Model not ready"}},
        )

    result = await engine.completion(request)

    if request.stream:
        from fastapi.responses import StreamingResponse

        async def stream_generator():
            async for chunk in result:
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return result


# Embeddings
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    engine = get_engine()
    if not engine.is_ready:
        return JSONResponse(
            status_code=503,
            content={"error": {"type": "service_unavailable", "message": "Model not ready"}},
        )

    return await engine.embeddings(request)


# Models
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models."""
    engine = get_engine()
    if not engine.is_ready:
        return JSONResponse(
            status_code=503,
            content={"error": {"type": "service_unavailable", "message": "Model not ready"}},
        )

    return await engine.list_models()


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str):
    """Get model information."""
    engine = get_engine()
    if not engine.is_ready:
        return JSONResponse(
            status_code=503,
            content={"error": {"type": "service_unavailable", "message": "Model not ready"}},
        )

    try:
        return await engine.get_model_info(model_id)
    except ValueError as e:
        return JSONResponse(
            status_code=404,
            content={"error": {"type": "not_found", "message": str(e)}},
        )


# Root
@app.get("/")
async def root():
    return {
        "service": "llm-inference-runtime",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics",
    }


def main():
    """Entry point for running the server."""
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        access_log=True,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()