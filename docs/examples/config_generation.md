---
description: Generate a Ludwig config from a natural language task description using an LLM, then train a model immediately — no YAML knowledge required.
---

# LLM Config Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/llm_config_generation/config_generation.ipynb)

Ludwig can generate a complete, validated YAML configuration from a plain-English description
of your ML task. This example shows how to go from a natural language description to a
trained model in under 20 lines of Python.

For the full user guide see [LLM Config Generation](../user_guide/llms/config_generation.md).

---

## Prerequisites

```bash
pip install ludwig anthropic  # or: pip install ludwig openai
export ANTHROPIC_API_KEY=sk-ant-...  # or OPENAI_API_KEY=sk-...
```

---

## Quickstart

```python
import yaml
from ludwig.config_generation import generate_config

# Describe your task in plain English
config = generate_config(
    "I have customer data with columns: age (integer), annual_income (float), "
    "num_purchases (integer), days_since_last_purchase (integer). "
    "I want to predict whether a customer will churn (binary 0 or 1)."
)

# Inspect the generated config
print(yaml.dump(config, default_flow_style=False, sort_keys=False))
```

Example output:
```yaml
model_type: ecd
input_features:
  - name: age
    type: number
    preprocessing:
      normalization: zscore
  - name: annual_income
    type: number
    preprocessing:
      normalization: zscore
  - name: num_purchases
    type: number
    preprocessing:
      normalization: zscore
  - name: days_since_last_purchase
    type: number
    preprocessing:
      normalization: zscore
output_features:
  - name: churn
    type: binary
combiner:
  type: concat
  num_fc_layers: 2
  output_size: 128
trainer:
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  optimizer:
    type: adamw
    weight_decay: 0.01
```

---

## Train on the generated config

```python
import pandas as pd
from ludwig.api import LudwigModel
from ludwig.config_generation import generate_config

config = generate_config(
    "I have customer data with columns: age, annual_income, "
    "num_purchases, days_since_last_purchase. "
    "I want to predict churn (binary)."
)

df = pd.read_csv("customers.csv")
model = LudwigModel(config=config)
results = model.train(dataset=df, output_directory="./results")

test_df = pd.read_csv("customers_test.csv")
predictions, _ = model.predict(dataset=test_df)
print(predictions.head())
```

---

## CLI usage

```bash
# Print to stdout
ludwig generate_config "predict house price from bedrooms, sqft, location, year_built"

# Save to file
ludwig generate_config "classify email as spam or ham" --output spam_config.yaml

# Use GPT-4o instead of Claude
ludwig generate_config --model gpt-4o "predict loan default from financial metrics"
```

---

## More examples

### Text classification

```python
config = generate_config(
    "I have product reviews (text column named 'review_text') and want to "
    "classify sentiment as positive, negative, or neutral (category)."
)
```

### Multimodal classification

```python
config = generate_config(
    "Classify real estate listings as luxury, mid-range, or budget based on: "
    "description (text), main_image (image path), num_bedrooms (number), "
    "square_feet (number), neighborhood (category)."
)
```

### Time series regression

```python
config = generate_config(
    "I have hourly energy consumption readings (number) and want to "
    "predict next-hour consumption. Features: hour_of_day (number), "
    "day_of_week (number), temperature (number), is_holiday (binary)."
)
```

---

## Customising the generated config

The generated config is a plain Python dict — edit it before training:

```python
config = generate_config("predict churn from age, income, num_purchases")

# Add more epochs and a learning rate scheduler
config["trainer"]["epochs"] = 100
config["trainer"]["learning_rate_scheduler"] = {
    "decay": "cosine",
    "warmup_fraction": 0.1,
}

# Switch to a transformer encoder for text features
for feat in config["input_features"]:
    if feat["type"] == "text":
        feat["encoder"] = {
            "type": "bert",
            "pretrained_model_name_or_path": "answerdotai/ModernBERT-base",
        }

model = LudwigModel(config=config)
```

---

## See also

- [LLM Config Generation user guide](../user_guide/llms/config_generation.md) — full API reference
- [Configuration reference](../configuration/index.md) — edit and extend the generated config
- [Hyperparameter optimization](hyperopt.md) — automatically tune the config
