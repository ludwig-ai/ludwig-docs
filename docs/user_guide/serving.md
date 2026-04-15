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
