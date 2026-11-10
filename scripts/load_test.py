#!/usr/bin/env python3
"""
Load test script for llm-inference-runtime.

Sustained load testing with configurable ramp-up, steady state, and ramp-down.
"""

import asyncio
import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
from datetime import datetime

import httpx


@dataclass
class LoadTestResult:
    timestamp: str
    phase: str
    concurrent_users: int
    requests_sent: int
    requests_completed: int
    requests_failed: int
    avg_latency_ms: float
    p95_latency_ms: float
    throughput_rps: float
    error_rate: float


class LoadTester:
    def __init__(
        self,
        url: str,
        model: str,
        prompt: str,
        max_tokens: int,
        api_key: str = "",
    ):
        self.url = url
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.results: List[LoadTestResult] = []
        self._running = False

    async def _make_request(self, client: httpx.AsyncClient) -> tuple[bool, float, int]:
        """Make a single request, return (success, latency, tokens)."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "stream": False,
        }

        start = time.perf_counter()
        try:
            response = await client.post(self.url, json=payload, headers=headers, timeout=60.0)
            latency = time.perf_counter() - start
            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                tokens = usage.get("completion_tokens", 0) + usage.get("prompt_tokens", 0)
                return True, latency, tokens
            return False, latency, 0
        except Exception:
            return False, time.perf_counter() - start, 0

    async def run_phase(
        self,
        client: httpx.AsyncClient,
        phase_name: str,
        concurrent: int,
        duration: int,
        ramp_up: float = 0,
    ) -> LoadTestResult:
        """Run a single load test phase."""
        print(f"\n[{phase_name}] Starting: {concurrent} concurrent users for {duration}s")
        if ramp_up > 0:
            print(f"  Ramp-up: {ramp_up}s")

        phase_start = time.time()
        results: List[tuple[bool, float, int]] = []
        semaphore = asyncio.Semaphore(concurrent)

        async def worker():
            async with semaphore:
                while self._running and (time.time() - phase_start) < duration:
                    results.append(await self._make_request(client))
                    await asyncio.sleep(0.01)  # Small delay to prevent overwhelming

        # Ramp up
        if ramp_up > 0:
            tasks = []
            for i in range(concurrent):
                if not self._running:
                    break
                tasks.append(asyncio.create_task(worker()))
                await asyncio.sleep(ramp_up / concurrent)
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            tasks = [asyncio.create_task(worker()) for _ in range(concurrent)]
            await asyncio.sleep(duration)
            self._running = False
            await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate metrics
        completed = [r for r in results if r[0]]
        failed = [r for r in results if not r[0]]
        latencies = [r[1] for r in completed]
        total_tokens = sum(r[2] for r in completed)

        phase_duration = time.time() - phase_start

        if latencies:
            latencies.sort()
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = latencies[int(len(latencies) * 0.95)]
        else:
            avg_latency = p95_latency = 0

        result = LoadTestResult(
            timestamp=datetime.utcnow().isoformat() + "Z",
            phase=phase_name,
            concurrent_users=concurrent,
            requests_sent=len(results),
            requests_completed=len(completed),
            requests_failed=len(failed),
            avg_latency_ms=avg_latency * 1000,
            p95_latency_ms=p95_latency * 1000,
            throughput_rps=len(completed) / phase_duration if phase_duration > 0 else 0,
            error_rate=len(failed) / len(results) if results else 0,
        )

        print(f"  Completed: {result.requests_completed}, Failed: {result.requests_failed}")
        print(f"  Avg Latency: {result.avg_latency_ms:.1f}ms, P95: {result.p95_latency_ms:.1f}ms")
        print(f"  Throughput: {result.throughput_rps:.1f} req/s, Error Rate: {result.error_rate*100:.1f}%")

        self.results.append(result)
        return result

    async def run(
        self,
        ramp_up_users: int,
        ramp_up_duration: int,
        steady_users: int,
        steady_duration: int,
        ramp_down_users: int,
        ramp_down_duration: int,
        output: str = "",
    ):
        """Run full load test with ramp-up, steady state, and ramp-down."""
        self._running = True

        print("=" * 60)
        print("LOAD TEST STARTING")
        print("=" * 60)
        print(f"URL: {self.url}")
        print(f"Model: {self.model}")
        print(f"Phases: Ramp-up({ramp_up_users}u/{ramp_up_duration}s) -> "
              f"Steady({steady_users}u/{steady_duration}s) -> "
              f"Ramp-down({ramp_down_users}u/{ramp_down_duration}s)")

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Warmup
            print("\nWarming up...")
            await self._make_request(client)

            # Ramp-up phase
            if ramp_up_users > 0 and ramp_up_duration > 0:
                await self.run_phase(
                    client, "ramp_up", ramp_up_users, ramp_up_duration, ramp_up=min(ramp_up_duration, 30)
                )

            # Steady state phase
            if steady_users > 0 and steady_duration > 0:
                await self.run_phase(
                    client, "steady", steady_users, steady_duration
                )

            # Ramp-down phase
            if ramp_down_users > 0 and ramp_down_duration > 0:
                await self.run_phase(
                    client, "ramp_down", ramp_down_users, ramp_down_duration
                )

        self._running = False

        # Print summary
        print("\n" + "=" * 60)
        print("LOAD TEST SUMMARY")
        print("=" * 60)
        total_completed = sum(r.requests_completed for r in self.results)
        total_failed = sum(r.requests_failed for r in self.results)
        total_sent = sum(r.requests_sent for r in self.results)
        avg_throughput = sum(r.throughput_rps for r in self.results) / len(self.results) if self.results else 0

        print(f"Total Requests Sent:     {total_sent}")
        print(f"Total Completed:         {total_completed}")
        print(f"Total Failed:            {total_failed}")
        print(f"Overall Error Rate:      {total_failed/total_sent*100:.1f}%" if total_sent > 0 else "N/A")
        print(f"Average Throughput:      {avg_throughput:.1f} req/s")

        # Save results
        if output:
            output_data = {
                "config": {
                    "url": self.url,
                    "model": self.model,
                    "ramp_up_users": ramp_up_users,
                    "ramp_up_duration": ramp_up_duration,
                    "steady_users": steady_users,
                    "steady_duration": steady_duration,
                    "ramp_down_users": ramp_down_users,
                    "ramp_down_duration": ramp_down_duration,
                },
                "results": [asdict(r) for r in self.results],
                "summary": {
                    "total_requests_sent": total_sent,
                    "total_completed": total_completed,
                    "total_failed": total_failed,
                    "overall_error_rate": total_failed / total_sent if total_sent > 0 else 0,
                    "average_throughput_rps": avg_throughput,
                }
            }
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {output}")


async def main():
    parser = argparse.ArgumentParser(description="Load test llm-inference-runtime")
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions", help="API endpoint")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-70B-Instruct", help="Model name")
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per request")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--output", default="", help="Output JSON file")

    # Ramp-up phase
    parser.add_argument("--ramp-up-users", type=int, default=10, help="Ramp-up concurrent users")
    parser.add_argument("--ramp-up-duration", type=int, default=60, help="Ramp-up duration (seconds)")

    # Steady state phase
    parser.add_argument("--steady-users", type=int, default=50, help="Steady state concurrent users")
    parser.add_argument("--steady-duration", type=int, default=300, help="Steady state duration (seconds)")

    # Ramp-down phase
    parser.add_argument("--ramp-down-users", type=int, default=10, help="Ramp-down concurrent users")
    parser.add_argument("--ramp-down-duration", type=int, default=60, help="Ramp-down duration (seconds)")

    args = parser.parse_args()

    tester = LoadTester(
        url=args.url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
    )

    await tester.run(
        ramp_up_users=args.ramp_up_users,
        ramp_up_duration=args.ramp_up_duration,
        steady_users=args.steady_users,
        steady_duration=args.steady_duration,
        ramp_down_users=args.ramp_down_users,
        ramp_down_duration=args.ramp_down_duration,
        output=args.output,
    )


if __name__ == "__main__":
    asyncio.run(main())