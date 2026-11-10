# Prometheus Metrics for llm-inference-runtime

import time
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from threading import Lock

from .config import get_settings


class MetricsCollector:
    """Thread-safe Prometheus metrics collector."""

    def __init__(self):
        self.settings = get_settings()
        self._lock = Lock()
        self._initialized = False

        # Request metrics
        self.requests_total = Counter(
            "llm_requests_total",
            "Total number of requests",
            ["model", "endpoint", "status"],
        )

        self.request_duration = Histogram(
            "llm_request_duration_seconds",
            "Request latency in seconds",
            ["model", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )

        self.tokens_total = Counter(
            "llm_tokens_total",
            "Total tokens processed",
            ["model", "type"],  # type: prompt, completion
        )

        self.active_requests = Gauge(
            "llm_active_requests",
            "Number of currently active requests",
            ["model"],
        )

        self.gpu_memory_used = Gauge(
            "llm_gpu_memory_used_bytes",
            "GPU memory used in bytes",
            ["model", "device"],
        )

        self.batch_size = Histogram(
            "llm_batch_size",
            "Batch sizes",
            ["model"],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
        )

        self.queue_wait = Histogram(
            "llm_queue_wait_seconds",
            "Time spent waiting in queue",
            ["model"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

        self.model_load_duration = Histogram(
            "llm_model_load_duration_seconds",
            "Model loading duration in seconds",
            ["model"],
            buckets=(1, 5, 10, 30, 60, 120, 300, 600),
        )

        self.cache_hits = Counter(
            "llm_cache_hits_total",
            "Cache hit/miss counts",
            ["model", "result"],  # result: hit, miss
        )

        self.rate_limit_rejected = Counter(
            "llm_rate_limit_rejected_total",
            "Requests rejected due to rate limiting",
            ["model", "key_hash"],
        )

    def record_request(
        self,
        model: str,
        endpoint: str,
        status: str,
        duration: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record a request completion."""
        with self._lock:
            self.requests_total.labels(model=model, endpoint=endpoint, status=status).inc()
            self.request_duration.labels(model=model, endpoint=endpoint).observe(duration)
            if prompt_tokens > 0:
                self.tokens_total.labels(model=model, type="prompt").inc(prompt_tokens)
            if completion_tokens > 0:
                self.tokens_total.labels(model=model, type="completion").inc(completion_tokens)

    def record_active_request(self, model: str, delta: int) -> None:
        """Increment/decrement active request count."""
        with self._lock:
            self.active_requests.labels(model=model).inc(delta)

    def record_gpu_memory(self, model: str, device: str, bytes_used: int) -> None:
        """Record GPU memory usage."""
        with self._lock:
            self.gpu_memory_used.labels(model=model, device=device).set(bytes_used)

    def record_batch_size(self, model: str, size: int) -> None:
        """Record batch size."""
        with self._lock:
            self.batch_size.labels(model=model).observe(size)

    def record_queue_wait(self, model: str, wait_seconds: float) -> None:
        """Record queue wait time."""
        with self._lock:
            self.queue_wait.labels(model=model).observe(wait_seconds)

    def record_model_load(self, model: str, duration: float) -> None:
        """Record model loading time."""
        with self._lock:
            self.model_load_duration.labels(model=model).observe(duration)

    def record_cache_result(self, model: str, hit: bool) -> None:
        """Record cache hit/miss."""
        with self._lock:
            self.cache_hits.labels(model=model, result="hit" if hit else "miss").inc()

    def record_rate_limit_rejected(self, model: str, key_hash: str) -> None:
        """Record rate limit rejection."""
        with self._lock:
            self.rate_limit_rejected.labels(model=model, key_hash=key_hash).inc()

    def generate_metrics(self) -> bytes:
        """Generate Prometheus metrics output."""
        return generate_latest()


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> Optional[MetricsCollector]:
    """Get the global metrics collector."""
    global _metrics
    if _metrics is None and get_settings().metrics_enabled:
        _metrics = MetricsCollector()
    return _metrics


def init_metrics() -> MetricsCollector:
    """Initialize the global metrics collector."""
    global _metrics
    _metrics = MetricsCollector()
    return _metrics