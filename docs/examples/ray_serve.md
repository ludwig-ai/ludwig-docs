---
description: Deploy Ludwig models at scale with Ray Serve (autoscaling multi-replica) or KServe (Kubernetes Open Inference Protocol v2) for production ML serving.
---

# Ray Serve and KServe Deployment

Ludwig models can be deployed to production with:

- **Ray Serve** — autoscaling multi-replica deployment on a Ray cluster
- **KServe** — Kubernetes-native serving via the Open Inference Protocol v2

Both expose the same `/predict` payload interface as the built-in FastAPI server, so your
client code works unchanged across local, Ray Serve, and KServe deployments.

For general serving options (local FastAPI, vLLM) see [Serving Ludwig Models](../user_guide/serving.md).

---

## Ray Serve

### Install

```bash
pip install "ludwig[distributed]"  # includes ray[serve]
```

### Deploy

=== "Python API"

    ```python
    import ray
    from ray import serve
    from ludwig.serve_ray_serve import deploy_ludwig_model

    ray.init(ignore_reinit_error=True)

    handle = deploy_ludwig_model(
        model_path="results/experiment_run/model",
        name="sentiment",
        num_replicas=2,
        ray_actor_options={"num_gpus": 1},  # omit for CPU
    )

    print("Deployment live at: http://localhost:8000/sentiment")
    ```

=== "CLI (deploy.py)"

    ```bash
    # Two GPU replicas on port 8080
    python examples/serving/ray_serve/deploy.py \
        --model_path ./results/my_model \
        --name sentiment \
        --num_replicas 2 \
        --gpu \
        --port 8080 \
        --block
    ```

### Send predictions

```bash
# Single record
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!", "stars": 5}'

# Batch (list of records)
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '[{"text": "great"}, {"text": "terrible"}]'
```

Python client:

```python
import httpx
import json

url = "http://localhost:8000/sentiment"

# Single record
response = httpx.post(url, json={"text": "Amazing product, highly recommend!"})
print(response.json())

# Batch
records = [{"text": "great"}, {"text": "terrible"}, {"text": "okay"}]
response = httpx.post(url, json=records)
print(response.json())
```

### Autoscaling

Ray Serve supports autoscaling out of the box. Modify `deploy_ludwig_model` to pass
`autoscaling_config`:

```python
from ray.serve.config import AutoscalingConfig
from ludwig.serve_ray_serve import make_ludwig_deployment_class

DeploymentClass = make_ludwig_deployment_class(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
)

# Autoscale between 1 and 10 replicas based on queue length
app = DeploymentClass.options(
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=10,
        target_num_ongoing_requests_per_replica=5,
    )
).bind("results/experiment_run/model")

serve.run(app, name="sentiment")
```

### Traffic splitting (A/B testing)

```python
from ray import serve
from ludwig.serve_ray_serve import make_ludwig_deployment_class

ModelV1 = make_ludwig_deployment_class(num_replicas=1)
ModelV2 = make_ludwig_deployment_class(num_replicas=1)

app_v1 = ModelV1.bind("results/model_v1")
app_v2 = ModelV2.bind("results/model_v2")

# Route 80% to v1, 20% to v2
serve.run(
    {"v1": serve.options(route_prefix="/v1")(app_v1),
     "v2": serve.options(route_prefix="/v2")(app_v2)},
)
```

---

## KServe

KServe implements the [Open Inference Protocol v2](https://kserve.github.io/website/modelserving/data_plane/v2_protocol/)
for Kubernetes-native ML serving.

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

Or with Python:

```python
from ludwig.serve_kserve import create_kserve_model_server

server = create_kserve_model_server(
    model_name="sentiment",
    model_path="results/experiment_run/model",
)
server.start([server.model])
```

### Send v2 protocol requests

```bash
# Health check
curl http://localhost:8080/v2/health/live

# Model metadata
curl http://localhost:8080/v2/models/sentiment

# Inference (v2 input format)
curl -X POST http://localhost:8080/v2/models/sentiment/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"name": "text", "shape": [1], "datatype": "BYTES", "data": ["I love this!"]},
      {"name": "stars", "shape": [1], "datatype": "INT32", "data": [5]}
    ]
  }'
```

### Deploy on Kubernetes with KServe

Create a `InferenceService` manifest:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: ludwig-sentiment
spec:
  predictor:
    containers:
      - name: predictor
        image: ludwigai/ludwig:latest
        command: ["python", "-m", "ludwig.serve_kserve"]
        args:
          - --model_name=sentiment
          - --model_path=/mnt/models
        volumeMounts:
          - name: model-volume
            mountPath: /mnt/models
    volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: ludwig-models-pvc
```

```bash
kubectl apply -f inference_service.yaml
kubectl get inferenceservice ludwig-sentiment
```

---

## Choosing between serving options

| Option | Scale | Protocol | Best for |
|--------|-------|----------|----------|
| `ludwig serve` (FastAPI) | Single process | REST JSON | Development, simple APIs |
| Ray Serve | Multi-replica, autoscaling | REST JSON | Production on Ray clusters |
| KServe | Kubernetes-native | OIP v2 + REST | Enterprise Kubernetes |
| `ludwig serve` (vLLM) | GPU-optimized | OpenAI-compatible | LLM serving |

---

## See also

- [Serving Ludwig Models](../user_guide/serving.md) — FastAPI and vLLM serving
- [Distributed training on Ray](../user_guide/distributed_training/index.md) — train on a Ray cluster
- [Model Export](../user_guide/model_export.md) — export to ONNX or TorchScript before serving
