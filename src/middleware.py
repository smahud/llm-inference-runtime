# Middleware for llm-inference-runtime

import time
import hashlib
import logging
from typing import Optional, Callable, Awaitable
from collections import defaultdict

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .config import get_settings
from .metrics import get_metrics

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """API Key authentication middleware."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
        self.valid_keys: set = set()
        self._parse_keys()

    def _parse_keys(self) -> None:
        """Parse API keys from settings."""
        if self.settings.api_keys:
            self.valid_keys = set(self.settings.api_keys)
            logger.info(f"API key authentication enabled with {len(self.valid_keys)} keys")
        else:
            logger.info("API key authentication disabled")

    def _extract_key(self, request: Request) -> Optional[str]:
        """Extract API key from request."""
        # Check Authorization header
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]

        # Check X-API-Key header
        return request.headers.get("X-API-Key")

    def _hash_key(self, key: str) -> str:
        """Hash key for logging/metrics (first 8 chars of SHA256)."""
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Skip auth for health/ready/metrics endpoints
        if request.url.path in ("/health", "/ready", "/metrics", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Skip if no keys configured
        if not self.valid_keys:
            return await call_next(request)

        # Extract and validate key
        api_key = self._extract_key(request)
        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": {"type": "auth_error", "message": "Missing API key"}},
                headers={"WWW-Authenticate": "Bearer"},
            )

        if api_key not in self.valid_keys:
            key_hash = self._hash_key(api_key)
            logger.warning(f"Invalid API key attempted: {key_hash}")

            metrics = get_metrics()
            if metrics:
                metrics.record_rate_limit_rejected(self.settings.model, key_hash)

            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": {"type": "auth_error", "message": "Invalid API key"}},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Store key hash in request state for rate limiting
        request.state.api_key_hash = self._hash_key(api_key)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiting middleware."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
        self.rpm_limit = self.settings.rate_limit_rpm
        self.tpm_limit = self.settings.rate_limit_tpm

        # Token buckets: key_hash -> {"rpm_tokens": float, "tpm_tokens": float, "last_refill": float}
        self.buckets: dict = defaultdict(lambda: {"rpm": float(self.rpm_limit), "tpm": float(self.tpm_limit), "last": time.time()})
        self._lock = None  # Will use asyncio.Lock in async context

    def _refill(self, bucket: dict, now: float) -> None:
        """Refill token buckets based on elapsed time."""
        elapsed = now - bucket["last"]
        if elapsed > 0:
            # Refill RPM (per minute)
            bucket["rpm"] = min(self.rpm_limit, bucket["rpm"] + elapsed * self.rpm_limit / 60.0)
            # Refill TPM (per minute)
            bucket["tpm"] = min(self.tpm_limit, bucket["tpm"] + elapsed * self.tpm_limit / 60.0)
            bucket["last"] = now

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Skip rate limiting for health/ready/metrics
        if request.url.path in ("/health", "/ready", "/metrics", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Skip if no auth (no key to rate limit)
        key_hash = getattr(request.state, "api_key_hash", None)
        if not key_hash:
            return await call_next(request)

        now = time.time()
        bucket = self.buckets[key_hash]
        self._refill(bucket, now)

        # Estimate tokens from request (rough approximation)
        estimated_tokens = 100  # Default estimate

        if bucket["rpm"] < 1 or bucket["tpm"] < estimated_tokens:
            metrics = get_metrics()
            if metrics:
                metrics.record_rate_limit_rejected(self.settings.model, key_hash)

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Rate limit exceeded",
                        "retry_after": 60,
                    }
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit-RPM": str(self.rpm_limit),
                    "X-RateLimit-Limit-TPM": str(self.tpm_limit),
                    "X-RateLimit-Remaining-RPM": "0",
                    "X-RateLimit-Remaining-TPM": "0",
                },
            )

        # Consume tokens
        bucket["rpm"] -= 1
        bucket["tpm"] -= estimated_tokens

        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit-RPM"] = str(self.rpm_limit)
        response.headers["X-RateLimit-Limit-TPM"] = str(self.tpm_limit)
        response.headers["X-RateLimit-Remaining-RPM"] = str(int(bucket["rpm"]))
        response.headers["X-RateLimit-Remaining-TPM"] = str(int(bucket["tpm"]))
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured request logging middleware."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        start_time = time.time()

        # Log request
        key_hash = getattr(request.state, "api_key_hash", "anonymous")
        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "key_hash": key_hash,
                "client": request.client.host if request.client else "unknown",
            },
        )

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            logger.info(
                "Request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "key_hash": key_hash,
                },
            )
            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration * 1000, 2),
                    "key_hash": key_hash,
                    "error": str(e),
                },
            )
            raise


class ActiveRequestMiddleware(BaseHTTPMiddleware):
    """Track active requests for metrics."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        metrics = get_metrics()
        if metrics:
            metrics.record_active_request(self.settings.model, 1)

        try:
            return await call_next(request)
        finally:
            if metrics:
                metrics.record_active_request(self.settings.model, -1)