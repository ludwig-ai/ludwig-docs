---
description: Fine-tune Vision-Language Models (VLMs) like Qwen2-VL and LLaVA on image+text tasks using Ludwig. Covers QLoRA, multimodal preprocessing, and VQA datasets.
---

# Vision-Language Model Fine-Tuning

Ludwig supports fine-tuning **Vision-Language Models (VLMs)** — pretrained models that jointly reason
over images and text. This lets you adapt models like Qwen2-VL, LLaVA, or Idefics to your own
image+text tasks without writing any custom training code.

## What is a VLM?

A Vision-Language Model combines a vision encoder (typically a ViT or CLIP-style model) with a
large language model backbone. The model receives both image patches and text tokens as input and
generates text as output. Common tasks include:

- **Visual Question Answering (VQA)** — answer natural-language questions about images
- **Image Captioning** — generate descriptions of images
- **Instruction following on images** — follow complex instructions that reference visual content
- **Document understanding** — parse charts, tables, or screenshots

## Prerequisites

```bash
pip install "ludwig[llm]"
```

To use models from HuggingFace Hub that require gated access (like Llama-based VLMs):

```bash
huggingface-cli login
```

## Dataset format

VLM fine-tuning requires a CSV with at least:

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | string | Absolute or relative path to an image file |
| `question` | string | Instruction or question about the image |
| `answer` | string | Expected response (fine-tuning target) |

```csv
image_path,question,answer
/data/dog.jpg,What breed is this dog?,Golden Retriever
/data/chart.png,What is the highest value on the y-axis?,42
```

## Configuration

### Minimal config (Qwen2-VL)

```yaml
model_type: llm
base_model: Qwen/Qwen2-VL-7B-Instruct

# Enable multimodal (VLM) mode — required for image+text models
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

# 4-bit quantisation to fit a 7B VLM on a single 24 GB GPU
quantization:
  bits: 4
  quantization_type: nf4
  compute_dtype: bfloat16

generation:
  max_new_tokens: 256
  temperature: 0.0
```

The key parameter is `is_multimodal: true`. When set, Ludwig:

1. Loads the model with `AutoModelForVision2Seq` instead of `AutoModelForCausalLM`
2. Loads the corresponding multimodal processor (`AutoProcessor`) for joint text+image tokenisation
3. Handles image patch extraction and interleaving with text tokens automatically

### Supported base models

Any HuggingFace `Vision2Seq` model works as a base model:

| Model | HuggingFace ID | Notes |
|-------|---------------|-------|
| Qwen2-VL 7B | `Qwen/Qwen2-VL-7B-Instruct` | Strong performance, requires `trust_remote_code: true` |
| Qwen2-VL 72B | `Qwen/Qwen2-VL-72B-Instruct` | Best quality, needs 2× A100 with QLoRA |
| LLaVA 1.5 | `llava-hf/llava-1.5-7b-hf` | Widely used baseline |
| LLaVA-NeXT | `llava-hf/llava-v1.6-mistral-7b-hf` | Improved resolution handling |
| Idefics-9B | `HuggingFaceM4/idefics-9b-instruct` | Multi-image support |
| InternVL2 | `OpenGVLab/InternVL2-8B` | Strong on document understanding |

## Training

```bash
ludwig train \
  --config vlm_config.yaml \
  --dataset vqa_dataset.csv \
  --output_directory ./results
```

Or via Python:

```python
from ludwig.api import LudwigModel

model = LudwigModel(config="vlm_config.yaml")
results = model.train(
    dataset="vqa_dataset.csv",
    output_directory="./results",
    skip_save_processed_input=True,  # images are large; skip caching
)
print(f"Model saved to: {results.output_directory}")
```

## Inference

After training, load the fine-tuned model and run predictions:

```python
import pandas as pd
from ludwig.api import LudwigModel

model = LudwigModel.load("results/experiment_run/model")

test_data = pd.DataFrame([
    {"image_path": "/data/test1.jpg", "question": "What color is the car?"},
    {"image_path": "/data/test2.jpg", "question": "How many people are in this image?"},
])
predictions, _ = model.predict(dataset=test_data)
print(predictions["answer_predictions"])
```

## Memory requirements

| Setup | GPU Memory | Recommended GPU |
|-------|-----------|-----------------|
| Qwen2-VL-7B, QLoRA 4-bit | ~16 GB | RTX 4090, A100-40GB |
| Qwen2-VL-7B, full fine-tune | ~48 GB | 2× A100-40GB |
| Qwen2-VL-72B, QLoRA 4-bit | ~80 GB | 2× A100-80GB |

## Distributed training on Ray

For large VLMs or large datasets, use Ray for multi-GPU training:

```yaml
backend:
  type: ray
  trainer:
    use_gpu: true
    num_workers: 2
    resources_per_worker:
      GPU: 1
```

```bash
ludwig train --config vlm_config.yaml --dataset vqa.csv
```

## See also

- [LLM Fine-Tuning guide](finetuning.md) — general LLM fine-tuning with LoRA/QLoRA
- [VLM fine-tuning example](../../examples/vlm_finetuning.md) — end-to-end walkthrough
- [Multi-adapter PEFT](multi_adapter.md) — merge multiple VLM adapters
- [LLM configuration reference](../../configuration/large_language_model.md)
