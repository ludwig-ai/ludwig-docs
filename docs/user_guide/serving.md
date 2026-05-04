# Serving Ludwig Models

Ludwig provides two serving options: a general-purpose REST API for ECD models and an OpenAI-compatible server for LLMs.

## REST API Server

### Starting the server

```bash
ludwig serve --model_path=/path/to/model
```

This spawns a FastAPI server with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/info` | GET | Model metadata (input/output features, model type) |
| `/predict` | POST | Single-record prediction |
| `/batch_predict` | POST | Batch prediction |
| `/metrics` | GET | Prometheus metrics (if prometheus_client is installed) |

### Single prediction

Send a JSON body with feature values:

```bash
curl http://0.0.0.0:8000/predict -X POST \
  -H "Content-Type: application/json" \
  -d '{"text_feature": "words to predict", "number_feature": 42}'
```

Response:
```json
{
  "output_predictions": "class_a",
  "output_probabilities_class_a": 0.82,
  "output_probabilities_class_b": 0.18,
  "output_probability": 0.82
}
```

### Batch prediction

Send a JSON array:

```bash
curl http://0.0.0.0:8000/batch_predict -X POST \
  -H "Content-Type: application/json" \
  -d '[{"text": "first example"}, {"text": "second example"}]'
```

### Features

- **Auto-generated Pydantic schemas** from model config for request/response validation
- **Prometheus metrics** at `/metrics` with request count and latency histograms
- **Structured logging** of every request (method, path, status, duration, client)
- **Configurable timeout** for long predictions (returns HTTP 504 on timeout)
- **Model hot-swap** via dependency injection

### Python API

```python
from ludwig.serve_v2 import create_app

app = create_app(
    model_path="path/to/model",
    allowed_origins=["*"],
    prediction_timeout=300.0,
)
# Use with uvicorn, gunicorn, or any ASGI server
```

The same timeout is exposed as the `--prediction_timeout` flag on `ludwig serve`. Requests
that exceed the timeout return HTTP 504 and are emitted through the structured-logging
middleware so they can be aggregated by downstream log collectors.

## LLM Serving with vLLM

For LLM models, Ludwig provides an OpenAI-compatible serving backend powered by vLLM with PagedAttention and continuous batching.

### Prerequisites

```bash
pip install vllm
```

### Starting the server

```python
from ludwig.serve_vllm import run_vllm_server

run_vllm_server(
    model_path="path/to/ludwig/model",
    host="0.0.0.0",
    port=8000,
    gpu_memory_utilization=0.9,
)
```

### OpenAI-compatible endpoints

The vLLM server exposes the same API as OpenAI's API:

```bash
# Chat completions
curl http://localhost:8000/v1/chat/completions -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ludwig-llm",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# Text completions
curl http://localhost:8000/v1/completions -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ludwig-llm",
    "prompt": "Machine learning is",
    "max_tokens": 100
  }'

# List models
curl http://localhost:8000/v1/models
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | required | Path to Ludwig model or HuggingFace model ID |
| `gpu_memory_utilization` | 0.9 | Fraction of GPU memory to use |
| `tensor_parallel_size` | 1 | Number of GPUs for tensor parallelism |
| `quantization` | None | Quantization method: awq, gptq, fp8 |
| `max_model_len` | auto | Maximum sequence length |

### Using with OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="ludwig-llm",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## Ray Serve

[Ray Serve](https://docs.ray.io/en/latest/serve/index.html) provides autoscaling multi-replica
serving on a Ray cluster. Use this for production deployments where you need horizontal scaling
or traffic splitting.

### Install

```bash
pip install "ludwig[distributed]"
```

### Deploy

```python
import ray
from ludwig.serve_ray_serve import deploy_ludwig_model

ray.init(ignore_reinit_error=True)

handle = deploy_ludwig_model(
    model_path="results/experiment_run/model",
    name="sentiment",
    num_replicas=2,
    ray_actor_options={"num_gpus": 1},  # omit for CPU
)
```

Or via the CLI helper:

```bash
python examples/serving/ray_serve/deploy.py \
    --model_path ./results/my_model \
    --num_replicas 2 \
    --gpu \
    --block
```

### Send predictions

```bash
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Batch (list of records)
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '[{"text": "great"}, {"text": "terrible"}]'
```

---

## KServe

[KServe](https://kserve.github.io) is the standard for Kubernetes ML inference, implementing
the Open Inference Protocol v2. Use this for cloud/Kubernetes production deployments.

### Install

```bash
pip install kserve
```

### Serve locally

```bash
python -m ludwig.serve_kserve \
    --model_name sentiment \
    --model_path results/experiment_run/model
```

### Send v2 protocol requests

```bash
curl -X POST http://localhost:8080/v2/models/sentiment/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"name": "text", "shape": [1], "datatype": "BYTES", "data": ["I love this!"]}
    ]
  }'
```

For full deployment examples (autoscaling, A/B testing, Kubernetes manifests) see the
[Ray Serve and KServe example](../examples/ray_serve.md).

---

## Choosing a serving option

| Option | Scale | Protocol | Best for |
|--------|-------|----------|----------|
| `ludwig serve` (FastAPI) | Single process | REST JSON | Development, simple APIs |
| Ray Serve | Multi-replica, autoscaling | REST JSON | Production on Ray clusters |
| KServe | Kubernetes-native | OIP v2 | Enterprise Kubernetes |
| `ludwig serve` (vLLM) | GPU-optimized | OpenAI-compatible | LLM serving |
