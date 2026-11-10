# Health and Readiness Checks for llm-inference-runtime

import time
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from .config import get_settings
from .engine import get_engine
from .metrics import get_metrics

router = APIRouter(tags=["health"])

# Track startup time
_start_time = time.time()


@dataclass
class HealthStatus:
    status: str = "unknown"
    checks: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HealthResponse(BaseModel):
    status: str
    service: str = "llm-inference-runtime"
    version: str = "1.0.0"
    uptime_seconds: int
    checks: Dict[str, Any]


class ReadyResponse(BaseModel):
    ready: bool
    service: str = "llm-inference-runtime"
    model_loaded: bool
    engine_ready: bool
    checks: Dict[str, Any]


async def check_engine() -> Dict[str, Any]:
    """Check if vLLM engine is initialized and responsive."""
    engine = get_engine()
    if not engine.is_ready:
        return {"status": "not_initialized", "healthy": False}

    try:
        # Quick health check - try to get model info
        await asyncio.wait_for(engine.list_models(), timeout=5.0)
        return {"status": "healthy", "healthy": True}
    except asyncio.TimeoutError:
        return {"status": "timeout", "healthy": False}
    except Exception as e:
        return {"status": "error", "healthy": False, "error": str(e)}


async def check_model_loaded() -> Dict[str, Any]:
    """Check if model is loaded and ready."""
    engine = get_engine()
    if not engine.is_ready:
        return {"status": "not_loaded", "loaded": False}

    try:
        model_info = await engine.get_model_info(get_settings().model)
        return {
            "status": "loaded",
            "loaded": True,
            "model_id": model_info.id,
            "max_model_len": getattr(model_info, "max_model_len", None),
        }
    except Exception as e:
        return {"status": "error", "loaded": False, "error": str(e)}


async def check_gpu() -> Dict[str, Any]:
    """Check GPU availability and memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"status": "no_cuda", "available": False}

        device_count = torch.cuda.device_count()
        devices = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = props.total_memory
            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(total / (1024**3), 2),
                "allocated_gb": round(allocated / (1024**3), 2),
                "reserved_gb": round(reserved / (1024**3), 2),
                "free_gb": round((total - reserved) / (1024**3), 2),
            })

        return {
            "status": "available",
            "available": True,
            "device_count": device_count,
            "devices": devices,
        }
    except ImportError:
        return {"status": "torch_not_installed", "available": False}
    except Exception as e:
        return {"status": "error", "available": False, "error": str(e)}


async def check_disk_space() -> Dict[str, Any]:
    """Check disk space for model cache."""
    import shutil
    settings = get_settings()
    try:
        total, used, free = shutil.disk_usage(settings.hf_home)
        return {
            "status": "ok" if free > 5 * (1024**3) else "low",
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def check_redis() -> Dict[str, Any]:
    """Check Redis connectivity if configured."""
    settings = get_settings()
    if not settings.redis_url:
        return {"status": "not_configured", "connected": False}

    try:
        import redis.asyncio as redis
        client = redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2)
        await client.ping()
        await client.close()
        return {"status": "connected", "connected": True}
    except Exception as e:
        return {"status": "error", "connected": False, "error": str(e)}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness probe - returns 200 if service is running.
    Does not check dependencies - just that the process is alive.
    """
    checks = {
        "process": {"status": "running", "healthy": True},
        "uptime_seconds": int(time.time() - _start_time),
    }

    # Add engine status if initialized
    engine = get_engine()
    if engine.is_ready:
        checks["engine"] = {"status": "initialized", "healthy": True}
    else:
        checks["engine"] = {"status": "initializing", "healthy": True}

    # Overall status
    all_healthy = all(
        check.get("healthy", True) if isinstance(check, dict) else True
        for check in checks.values()
    )

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        uptime_seconds=int(time.time() - _start_time),
        checks=checks,
    )


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """
    Readiness probe - returns 200 only if ready to serve traffic.
    Checks: engine initialized, model loaded, GPU available.
    """
    checks = {}

    # Run all checks in parallel
    engine_check, model_check, gpu_check, disk_check, redis_check = await asyncio.gather(
        check_engine(),
        check_model_loaded(),
        check_gpu(),
        check_disk_space(),
        check_redis(),
        return_exceptions=True,
    )

    checks["engine"] = engine_check if not isinstance(engine_check, Exception) else {"status": "error", "healthy": False}
    checks["model"] = model_check if not isinstance(model_check, Exception) else {"status": "error", "loaded": False}
    checks["gpu"] = gpu_check if not isinstance(gpu_check, Exception) else {"status": "error", "available": False}
    checks["disk"] = disk_check if not isinstance(disk_check, Exception) else {"status": "error"}
    checks["redis"] = redis_check if not isinstance(redis_check, Exception) else {"status": "error", "connected": False}

    # Determine readiness
    engine_ready = checks["engine"].get("healthy", False)
    model_loaded = checks["model"].get("loaded", False)
    gpu_available = checks["gpu"].get("available", False)

    ready = engine_ready and model_loaded and gpu_available

    return ReadyResponse(
        ready=ready,
        model_loaded=model_loaded,
        engine_ready=engine_ready,
        checks=checks,
    )


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    from fastapi import Response
    metrics_collector = get_metrics()
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics not enabled")

    return Response(
        content=metrics_collector.generate_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/version")
async def version() -> Dict[str, str]:
    """Version information."""
    return {
        "service": "llm-inference-runtime",
        "version": "1.0.0",
        "build_time": "",  # Could be injected at build time
        "git_commit": "",  # Could be injected at build time
    }