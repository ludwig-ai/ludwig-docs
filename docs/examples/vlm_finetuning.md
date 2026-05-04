---
description: End-to-end example of fine-tuning a Vision-Language Model (VLM) like Qwen2-VL on a visual question answering dataset using Ludwig and QLoRA.
---

# VLM Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/vlm_finetuning/vlm_finetuning.ipynb)

This example fine-tunes **Qwen2-VL-7B-Instruct** on a Visual Question Answering (VQA) task.
The model learns to answer natural-language questions about images using 4-bit QLoRA, so it
fits on a single 24 GB GPU.

For the full user guide see [VLM Fine-Tuning](../user_guide/llms/vlm_finetuning.md).

---

## Dataset

The example uses a VQA dataset formatted as a CSV with three columns:

| Column | Description |
|--------|-------------|
| `image_path` | Path to image file |
| `question` | Natural-language question about the image |
| `answer` | Expected answer (fine-tuning target) |

```csv
image_path,question,answer
/data/dog.jpg,What breed is this dog?,Golden Retriever
/data/chart.png,What is the peak value in the chart?,42
/data/receipt.jpg,What is the total amount?,$ 18.50
```

### Using a HuggingFace VQA dataset

You can also use datasets from HuggingFace:

```python
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Load VQA-v2 (small split for demo)
ds = load_dataset("HuggingFaceM4/VQAv2", split="validation[:2000]", trust_remote_code=True)

# Save images locally and build CSV
img_dir = Path("vqa_images")
img_dir.mkdir(exist_ok=True)

rows = []
for i, item in enumerate(ds):
    img_path = img_dir / f"{i}.jpg"
    item["image"].save(img_path)
    rows.append({
        "image_path": str(img_path),
        "question": item["question"],
        "answer": item["multiple_choice_answer"],
    })

df = pd.DataFrame(rows)
df.to_csv("vqa_dataset.csv", index=False)
```

---

## Configuration

```yaml
# vlm_config.yaml
model_type: llm
base_model: Qwen/Qwen2-VL-7B-Instruct

is_multimodal: true
trust_remote_code: true

input_features:
  - name: image_path
    type: image
  - name: question
    type: text

output_features:
  - name: answer
    type: text

adapter:
  type: lora
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj"]

trainer:
  type: finetune
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  learning_rate_scheduler:
    decay: cosine
    warmup_fraction: 0.03

quantization:
  bits: 4
  quantization_type: nf4
  compute_dtype: bfloat16

generation:
  max_new_tokens: 256
  temperature: 0.0
```

---

## Training

```bash
pip install "ludwig[llm]"
huggingface-cli login  # needed for gated models

ludwig train \
  --config vlm_config.yaml \
  --dataset vqa_dataset.csv \
  --output_directory ./results
```

Or with the Python API:

```python
from ludwig.api import LudwigModel

model = LudwigModel(config="vlm_config.yaml")
results = model.train(
    dataset="vqa_dataset.csv",
    output_directory="./results",
    skip_save_processed_input=True,
)
print(f"Model saved to: {results.output_directory}")
```

---

## Inference

```python
import pandas as pd
from ludwig.api import LudwigModel

model = LudwigModel.load("results/experiment_run/model")

questions = pd.DataFrame([
    {"image_path": "test1.jpg", "question": "What color is the car?"},
    {"image_path": "test2.jpg", "question": "How many people are visible?"},
])

predictions, _ = model.predict(dataset=questions)
for q, a in zip(questions["question"], predictions["answer_predictions"]):
    print(f"Q: {q}")
    print(f"A: {a}\n")
```

---

## Multi-GPU training

For larger models or datasets, scale to multiple GPUs with Ray:

```yaml
# vlm_config_distributed.yaml — append to the config above
backend:
  type: ray
  trainer:
    use_gpu: true
    num_workers: 4
    resources_per_worker:
      GPU: 1
```

```bash
ray start --head
ludwig train --config vlm_config_distributed.yaml --dataset vqa_dataset.csv
```

---

## Alternative VLM base models

Swap `base_model` for any HuggingFace `Vision2Seq` model:

```yaml
# LLaVA 1.5
base_model: llava-hf/llava-1.5-7b-hf
is_multimodal: true
trust_remote_code: false  # LLaVA doesn't need this

# InternVL2 (strong on document tasks)
base_model: OpenGVLab/InternVL2-8B
is_multimodal: true
trust_remote_code: true
```

---

## See also

- [VLM Fine-Tuning user guide](../user_guide/llms/vlm_finetuning.md) — full reference
- [LLM Fine-Tuning](llms/llm_finetuning.md) — text-only LLM fine-tuning
- [Multi-adapter PEFT](../user_guide/llms/multi_adapter.md) — merge specialized VLM adapters
