---
description: Generate Ludwig YAML configs from plain English task descriptions using an LLM. Describe your ML task and get a validated config you can train immediately.
---

# LLM Config Generation

Ludwig can generate a complete, validated YAML configuration from a plain-English description of
your machine learning task. This is useful for rapid prototyping, onboarding new users, or
automating ML pipeline setup.

## How it works

`ludwig.config_generation.generate_config` sends your task description to Claude or GPT-4, along
with a compact representation of Ludwig's configuration schema. The LLM returns a JSON config
that is then validated with full Pydantic validation before being returned to you.

## Prerequisites

Install an LLM provider SDK:

```bash
pip install anthropic    # for Claude (default)
# or
pip install openai       # for GPT-4o
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

## Python API

```python
from ludwig.config_generation import generate_config

config = generate_config(
    "I have a CSV with columns: age (number), annual_income (number), "
    "num_purchases (integer), and days_since_last_purchase (integer). "
    "I want to predict whether a customer will churn (binary: 0 or 1)."
)

# config is a validated dict you can pass directly to LudwigModel
from ludwig.api import LudwigModel
model = LudwigModel(config=config)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_description` | — | Plain-English description of your ML task |
| `model` | `claude-sonnet-4-20250514` | LLM to use. Claude models start with `claude-`, OpenAI with `gpt-` |
| `api_key` | `None` | API key; defaults to reading `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` env vars |
| `validate` | `True` | Whether to validate the generated config with Pydantic before returning |

### Example descriptions

**Tabular classification:**
```python
config = generate_config(
    "Predict loan default from: age (number), income (number), "
    "debt_to_income_ratio (number), loan_purpose (category), "
    "credit_score (number). Target: defaulted (binary)."
)
```

**Text classification:**
```python
config = generate_config(
    "I have product reviews (text) and want to classify sentiment "
    "as positive, negative, or neutral (category)."
)
```

**Multi-output regression:**
```python
config = generate_config(
    "Given wine chemical properties (fixed_acidity, volatile_acidity, "
    "citric_acid, pH — all numbers), predict quality_score (number 1-10) "
    "and whether the wine is good (binary)."
)
```

**Multimodal:**
```python
config = generate_config(
    "I have a dataset of product listings with: title (text), "
    "image_path (image), and price (number). "
    "Predict whether the listing will sell within 7 days (binary)."
)
```

## CLI

```bash
ludwig generate_config "predict house price from bedrooms, sqft, location"
```

Save to file:
```bash
ludwig generate_config "classify email as spam or ham" --output config.yaml
```

Use GPT-4o:
```bash
ludwig generate_config --model gpt-4o "predict churn from user activity metrics"
```

## Inspecting and editing the generated config

Always review the generated config before training. Print it as YAML:

```python
import yaml
from ludwig.config_generation import generate_config

config = generate_config("predict house price from bedrooms, sqft, location, year_built")
print(yaml.dump(config, default_flow_style=False, sort_keys=False))
```

The generated config is a standard Ludwig config dict — edit it freely before passing to
`LudwigModel`.

## Training on the generated config

```python
import pandas as pd
from ludwig.api import LudwigModel
from ludwig.config_generation import generate_config

config = generate_config(
    "predict customer churn from age, income, num_purchases, days_since_last_purchase"
)

df = pd.read_csv("customers.csv")
model = LudwigModel(config=config)
results = model.train(dataset=df, output_directory="./results")
```

## Limitations

- Generated configs may need adjustment for domain-specific encoders (e.g., specifying
  `pretrained_model_name_or_path` for text encoders).
- LLM-generated configs are validated but not guaranteed to be optimal — treat them as a
  starting point for hyperparameter tuning.
- Image, audio, and time series features require correct file paths in the dataset.
- Very complex configs (multi-adapter PEFT, custom backends) may require manual editing.

## See also

- [Configuration reference](../../configuration/index.md) — full YAML schema
- [LLM Config Generation example](../../examples/config_generation.md) — runnable walkthrough
- [Hyperparameter optimization](../../user_guide/hyperopt.md) — tune the generated config
