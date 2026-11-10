#!/usr/bin/env python3
"""
Benchmark script for llm-inference-runtime.

Measures throughput, latency, and token generation performance.
"""

import asyncio
import argparse
import json
import statistics
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List

import httpx


@dataclass
class BenchmarkResult:
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    requests_per_second: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_mean: float
    total_tokens: int
    tokens_per_second: float


async def benchmark_chat(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrent: int,
    total_requests: int,
    api_key: str = "",
) -> BenchmarkResult:
    """Run chat completion benchmark."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }

    semaphore = asyncio.Semaphore(concurrent)
    latencies: List[float] = []
    total_tokens = 0
    successful = 0
    failed = 0

    async def make_request():
        nonlocal total_tokens, successful, failed
        async with semaphore:
            start = time.perf_counter()
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=60.0)
                latency = time.perf_counter() - start
                if response.status_code == 200:
                    data = response.json()
                    latencies.append(latency)
                    usage = data.get("usage", {})
                    total_tokens += usage.get("completion_tokens", 0) + usage.get("prompt_tokens", 0)
                    successful += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

    start_time = time.perf_counter()
    tasks = [make_request() for _ in range(total_requests)]
    await asyncio.gather(*tasks)
    total_duration = time.perf_counter() - start_time

    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        mean = statistics.mean(latencies)
    else:
        p50 = p95 = p99 = mean = 0.0

    return BenchmarkResult(
        total_requests=total_requests,
        successful_requests=successful,
        failed_requests=failed,
        total_duration=total_duration,
        requests_per_second=successful / total_duration if total_duration > 0 else 0,
        latency_p50=p50,
        latency_p95=p95,
        latency_p99=p99,
        latency_mean=mean,
        total_tokens=total_tokens,
        tokens_per_second=total_tokens / total_duration if total_duration > 0 else 0,
    )


async def main():
    parser = argparse.ArgumentParser(description="Benchmark llm-inference-runtime")
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions", help="API endpoint")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-70B-Instruct", help="Model name")
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per request")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--total", type=int, default=100, help="Total requests")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--output", default="", help="Output JSON file")
    args = parser.parse_args()

    print(f"Benchmarking {args.url}")
    print(f"Model: {args.model}")
    print(f"Concurrent: {args.concurrent}, Total: {args.total}")
    print("-" * 50)

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Warmup
        print("Warming up...")
        await client.post(args.url, json={
            "model": args.model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        }, headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}, timeout=30.0)

        # Run benchmark
        print("Running benchmark...")
        result = await benchmark_chat(
            client=client,
            url=args.url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            concurrent=args.concurrent,
            total_requests=args.total,
            api_key=args.api_key,
        )

    # Print results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total Requests:     {result.total_requests}")
    print(f"Successful:         {result.successful_requests}")
    print(f"Failed:             {result.failed_requests}")
    print(f"Total Duration:     {result.total_duration:.2f}s")
    print(f"Requests/sec:       {result.requests_per_second:.2f}")
    print(f"Latency (mean):     {result.latency_mean*1000:.1f}ms")
    print(f"Latency (p50):      {result.latency_p50*1000:.1f}ms")
    print(f"Latency (p95):      {result.latency_p95*1000:.1f}ms")
    print(f"Latency (p99):      {result.latency_p99*1000:.1f}ms")
    print(f"Total Tokens:       {result.total_tokens}")
    print(f"Tokens/sec:         {result.tokens_per_second:.1f}")
    print("=" * 50)

    # Save to file
    if args.output:
        output_data = {
            "config": vars(args),
            "results": {
                "total_requests": result.total_requests,
                "successful_requests": result.successful_requests,
                "failed_requests": result.failed_requests,
                "total_duration": result.total_duration,
                "requests_per_second": result.requests_per_second,
                "latency_p50_ms": result.latency_p50 * 1000,
                "latency_p95_ms": result.latency_p95 * 1000,
                "latency_p99_ms": result.latency_p99 * 1000,
                "latency_mean_ms": result.latency_mean * 1000,
                "total_tokens": result.total_tokens,
                "tokens_per_second": result.tokens_per_second,
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())