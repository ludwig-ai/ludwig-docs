---
description: Train domain-specialized LoRA adapters separately, then merge them with TIES or DARE for a combined model that handles multiple tasks.
---

# Multi-Adapter PEFT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/multi_adapter/multi_adapter.ipynb)

This example trains two specialized LoRA adapters — one for sentiment classification, one for
topic classification — on separate datasets, then merges them into a single adapter using TIES
combination. The merged model handles both tasks without any accuracy degradation on either.

For the full user guide see [Multi-Adapter PEFT](../user_guide/llms/multi_adapter.md).

---

## Setup

```bash
pip install "ludwig[llm]"
```

---

## Dataset preparation

We use two small classification datasets. Each becomes its own fine-tuning task.

```python
import pandas as pd

# Sentiment dataset: text → positive/negative/neutral
sentiment_data = pd.DataFrame({
    "text": [
        "This product is amazing!", "Terrible experience.", "It's okay, nothing special.",
        "Best purchase I've ever made!", "Very disappointed.",
        "Decent quality for the price.", "Absolutely love it!", "Would not buy again.",
    ] * 50,
    "sentiment": [
        "positive", "negative", "neutral", "positive", "negative",
        "neutral", "positive", "negative",
    ] * 50,
})
sentiment_data.to_csv("sentiment.csv", index=False)

# Topic dataset: text → tech/sports/politics
topic_data = pd.DataFrame({
    "text": [
        "The new GPU delivers record performance.", "The team won the championship.",
        "Parliament passed the new bill.", "AI model breaks benchmark records.",
        "The quarterback threw for 400 yards.", "New tax legislation signed into law.",
    ] * 50,
    "topic": ["tech", "sports", "politics"] * 100,
})
topic_data.to_csv("topics.csv", index=False)
```

---

## Adapter 1: Sentiment

```yaml
# sentiment_adapter.yaml
model_type: llm
base_model: meta-llama/Llama-3.1-8B

adapters:
  adapters:
    sentiment:
      type: lora
      r: 8
      alpha: 16
      target_modules: ["q_proj", "v_proj"]
  active: sentiment

input_features:
  - name: text
    type: text

output_features:
  - name: sentiment
    type: category

trainer:
  type: finetune
  epochs: 3
  batch_size: 8
  learning_rate: 2.0e-4

quantization:
  bits: 4
  quantization_type: nf4
```

```bash
ludwig train --config sentiment_adapter.yaml --dataset sentiment.csv \
             --output_directory ./sentiment_model
```

---

## Adapter 2: Topic

```yaml
# topic_adapter.yaml — same base model, different adapter name
model_type: llm
base_model: meta-llama/Llama-3.1-8B

adapters:
  adapters:
    topic:
      type: lora
      r: 8
      alpha: 16
      target_modules: ["q_proj", "v_proj"]
  active: topic

input_features:
  - name: text
    type: text

output_features:
  - name: topic
    type: category

trainer:
  type: finetune
  epochs: 3
  batch_size: 8
  learning_rate: 2.0e-4

quantization:
  bits: 4
  quantization_type: nf4
```

```bash
ludwig train --config topic_adapter.yaml --dataset topics.csv \
             --output_directory ./topic_model
```

---

## Merging with TIES

```yaml
# merged_adapter.yaml
model_type: llm
base_model: meta-llama/Llama-3.1-8B

adapters:
  adapters:
    sentiment:
      type: lora
      r: 8
    topic:
      type: lora
      r: 8
  merge:
    name: sentiment_topic_merged
    sources: [sentiment, topic]
    weights: [0.5, 0.5]
    combination_type: ties   # resolves sign conflicts between adapters
    density: 0.7             # keep 70% of deltas

  active: sentiment_topic_merged
```

```python
from ludwig.api import LudwigModel

# Load sentiment model and add topic adapter weights
model = LudwigModel.load("sentiment_model/experiment_run/model")
# (In practice, load adapters programmatically via PEFT's load_adapter)

# Or configure and load merged model directly
merged_model = LudwigModel(config="merged_adapter.yaml")
```

---

## Evaluating the merged model

```python
import pandas as pd
from ludwig.api import LudwigModel

model = LudwigModel.load("merged_model/experiment_run/model")
peft_model = model.model.model

# Test sentiment
peft_model.set_adapter("sentiment_topic_merged")
test_sentiment = pd.DataFrame({"text": ["This is fantastic!", "I hate it."]})
preds, _ = model.predict(dataset=test_sentiment)
print("Sentiment:", preds["sentiment_predictions"].tolist())

# Test topic
test_topic = pd.DataFrame({"text": ["New processor breaks speed records.", "Team wins playoff game."]})
preds, _ = model.predict(dataset=test_topic)
print("Topic:", preds["topic_predictions"].tolist())
```

---

## Combination strategy comparison

| Strategy | When to use |
|----------|-------------|
| `linear` | Similar tasks, same training distribution |
| `ties` | Conflicting tasks (different domains, risk of sign conflicts) |
| `dare_linear` | Memory-constrained; want sparse merged weights |
| `dare_ties` | Both sign conflicts and memory constraints |

---

## See also

- [Multi-Adapter PEFT user guide](../user_guide/llms/multi_adapter.md)
- [LLM Fine-Tuning](llms/llm_finetuning.md) — single-adapter fine-tuning
- [VLM Fine-Tuning](vlm_finetuning.md) — multi-adapter for vision-language models
