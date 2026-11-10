# llm-inference-runtime - Production-grade LLM Inference Gateway

High-throughput, multi-GPU LLM inference gateway with dynamic batching, vLLM engine integration, and token streaming.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  API Gateway │────▶│  vLLM Pool  │
│  (OpenAI    │     │  (FastAPI)   │     │  (Multi-GPU)│
│  Compatible)│     │  + Router    │     │  + Scheduler│
└─────────────┘     └──────────────┘     └─────────────┘
                           │                       │
                    ┌──────┴──────┐         ┌─────┴─────┐
                    │  Redis      │         │  Prometheus│
                    │  (Cache/    │         │  (Metrics)│
                    │   Queue)    │         └───────────┘
                    └─────────────┘
```

## Features

- **Dynamic Batching**: Continuous batching with configurable max batch size
- **Multi-GPU Support**: Tensor parallelism + pipeline parallelism via vLLM
- **OpenAI Compatible**: Drop-in replacement for `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- **Streaming**: SSE token streaming with configurable chunk size
- **Observability**: Prometheus metrics, structured logging, request tracing
- **Autoscaling Ready**: Horizontal pod autoscaler metrics exposed
- **Model Management**: Hot-swap models, multi-model serving
- **Authentication**: API key validation, rate limiting
- **Graceful Shutdown**: In-flight request completion

## Quick Start

```bash
# Build image
docker build -t ghcr.io/smahud/llm-inference-runtime:latest .

# Run single GPU
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL=meta-llama/Meta-Llama-3-70B-Instruct \
  ghcr.io/smahud/llm-inference-runtime:latest

# Run multi-GPU (tensor parallel)
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL=meta-llama/Meta-Llama-3-70B-Instruct \
  -e TENSOR_PARALLEL_SIZE=4 \
  ghcr.io/smahud/llm-inference-runtime:latest
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `MODEL` | Required | HuggingFace model ID or local path |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `PIPELINE_PARALLEL_SIZE` | `1` | Number of GPUs for pipeline parallelism |
| `MAX_BATCH_SIZE` | `256` | Max tokens per batch |
| `MAX_SEQ_LEN` | `4096` | Max sequence length |
| `MAX_NUM_SEQS` | `256` | Max concurrent sequences |
| `PORT` | `8000` | HTTP port |
| `WORKERS` | `1` | Uvicorn workers (use 1 for vLLM) |
| `API_KEYS` | "" | Comma-separated API keys (optional) |
| `RATE_LIMIT_RPM` | `1000` | Requests per minute per key |
| `LOG_LEVEL` | `info` | Logging level |
| `TRUST_REMOTE_CODE` | `false` | Allow remote code execution |
| `DTYPE` | `auto` | Model dtype (auto, half, bfloat16, float16) |
| `QUANTIZATION` | `none` | Quantization method (none, awq, gptq, sq) |
| `GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of GPU memory to use |
| `SWAP_SPACE` | `4` | CPU swap space in GB |
| `DISABLE_LOG_STATS` | `false` | Disable vLLM stats logging |
| `ENABLE_CHUNKED_PREFILL` | `true` | Enable chunked prefill for long contexts |
| `ENABLE_PREFIX_CACHING` | `true` | Enable automatic prefix caching |

## API Endpoints

### Chat Completions
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-70B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### Completions (Legacy)
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Meta-Llama-3-70B-Instruct", "prompt": "Once upon a time", "max_tokens": 100}'
```

### Embeddings
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Meta-Llama-3-70B-Instruct", "input": "Hello world"}'
```

### Models
```bash
curl http://localhost:8000/v1/models
```

### Health
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### Metrics
```bash
curl http://localhost:8000/metrics
```

## Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference-runtime
  labels:
    app: llm-inference-runtime
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-inference-runtime
  template:
    metadata:
      labels:
        app: llm-inference-runtime
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: inference
        image: ghcr.io/smahud/llm-inference-runtime:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL
          value: "meta-llama/Meta-Llama-3-70B-Instruct"
        - name: TENSOR_PARALLEL_SIZE
          value: "4"
        - name: GPU_MEMORY_UTILIZATION
          value: "0.9"
        resources:
          limits:
            nvidia.com/gpu: 4
          requests:
            nvidia.com/gpu: 4
            memory: "80Gi"
            cpu: "16"
        volumeMounts:
        - name: hf-cache
          mountPath: /root/.cache/huggingface
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: hf-cache
        persistentVolumeClaim:
          claimName: hf-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llm-inference-runtime
spec:
  selector:
    app: llm-inference-runtime
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

## Monitoring

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `llm_requests_total` | Counter | Total requests by model, endpoint, status |
| `llm_request_duration_seconds` | Histogram | Request latency |
| `llm_tokens_total` | Counter | Total tokens (prompt + completion) |
| `llm_active_requests` | Gauge | Currently processing requests |
| `llm_gpu_memory_used_bytes` | Gauge | GPU memory usage |
| `llm_batch_size` | Histogram | Batch sizes |
| `llm_queue_wait_seconds` | Histogram | Time waiting in queue |

### Grafana Dashboard

Import `k8s/grafana-dashboard.json` for pre-built dashboards.

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run locally (requires GPU)
MODEL=meta-llama/Meta-Llama-3-8B-Instruct python -m src.main

# Format code
ruff format src/ tests/
ruff check src/ tests/
```

## Project Structure

```
llm-inference-runtime/
├── src/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration management
│   ├── engine.py               # vLLM engine wrapper
│   ├── router.py               # Request routing & batching
│   ├── models.py               # Pydantic models
│   ├── metrics.py              # Prometheus metrics
│   ├── middleware.py           # Auth, rate limiting, logging
│   └── health.py               # Health/readiness checks
├── tests/
│   ├── test_health.py
│   ├── test_completions.py
│   ├── test_streaming.py
│   ├── test_metrics.py
│   └── conftest.py
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── pvc.yaml
│   └── grafana-dashboard.json
├── config/
│   ├── logging.yaml
│   └── prometheus.yml
├── scripts/
│   ├── benchmark.py
│   ├── load_test.py
│   └── model_downloader.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

## Topics

`python` `vllm` `llmops` `triton` `grpc` `inference` `llm` `kubernetes`