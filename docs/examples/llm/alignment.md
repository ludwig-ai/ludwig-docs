# LLM Alignment with Preference Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/alignment/alignment_dpo.ipynb)

Alignment training adapts a language model so its outputs match human values and preferences.
The classic RLHF pipeline — collect human rankings, train a reward model, run PPO — is expensive
and notoriously unstable. Ludwig provides a family of modern preference learning trainers that
skip the reward model entirely and can be run with the same `ludwig train` command used for
standard fine-tuning.

This page covers:

- When to use each alignment method
- How to prepare datasets for each trainer
- CLI and Python API examples
- Uploading the aligned model to HuggingFace Hub

The full runnable notebook is in
[`examples/alignment/alignment_dpo.ipynb`](https://github.com/ludwig-ai/ludwig/blob/main/examples/alignment/alignment_dpo.ipynb).

## Alignment trainers

Ludwig supports four preference learning trainers, all accessible via `trainer.type`:

| Trainer | Paper | Data format | Key hyperparameter |
|---------|-------|-------------|-------------------|
| `dpo` | [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) | `prompt`, `chosen`, `rejected` | `beta` — KL penalty (default 0.1) |
| `kto` | [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306) | `prompt`, `response`, `label` (bool) | `desirable_weight`, `undesirable_weight` |
| `orpo` | [Hong et al., 2024](https://arxiv.org/abs/2403.07691) | `prompt`, `chosen`, `rejected` | — (no reference model) |
| `grpo` | [Shao et al., 2024](https://arxiv.org/abs/2402.03300) | `prompt` + reward function | `num_generations` — rollouts per prompt |

### DPO

**Direct Preference Optimization** reformulates the RLHF objective so the policy is the only
learned parameter — no reward model is required. The loss increases the log-probability of the
chosen response relative to the rejected response, subject to a KL-divergence penalty controlled
by `beta`.

DPO is the most widely studied alignment method and a natural first choice when you have
paired preference data.

### KTO

**Kahneman-Tversky Optimization** draws on prospect theory: humans are more sensitive to losses
than to equivalent gains. KTO uses this insight to define a loss that works with single-label
feedback — each response is simply marked as desirable (`label=True`) or undesirable (`label=False`).

This makes KTO well-suited when collecting binary user feedback (thumbs up/down, click-through
rates) is easier than asking annotators to compare two responses head-to-head.

### ORPO

**Odds Ratio Preference Optimization** combines supervised fine-tuning and preference alignment
in a single training objective. Unlike DPO, ORPO does not require a separate reference model
forward pass, which reduces GPU memory and compute by roughly half. Use ORPO when you want to
skip the SFT stage and align in one shot.

### GRPO

**Group Relative Policy Optimization** is a reinforcement learning trainer that generates
multiple completions per prompt, scores them with a reward function, and trains the policy
using the group-normalised advantage. Unlike DPO and KTO, GRPO does not require pre-collected
human annotations — the reward function can be entirely programmatic (e.g. pass/fail unit tests
for code generation, exact-match for math).

## Dataset preparation

### DPO and ORPO

Your dataset must contain `prompt`, `chosen`, and `rejected` columns.

```
prompt,chosen,rejected
"Explain gravity in simple terms","Gravity is a force that pulls objects with mass toward each other...","IDK lol"
"Write a haiku about autumn","Leaves drift silently / Crimson and gold fill the air / Winter waits below","autumn is nice i guess"
```

The [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset is a common
starting point. The `prepare_dataset.py` script in the
[examples/alignment](https://github.com/ludwig-ai/ludwig/blob/main/examples/alignment/)
directory downloads it and converts it automatically:

```bash
python examples/alignment/prepare_dataset.py --output_dir data/ --max_train_samples 10000
```

To do the conversion manually:

```python
import re
import pandas as pd
from datasets import load_dataset


def last_human_turn(conv):
    turns = re.findall(r"\n\nHuman: (.*?)(?=\n\nAssistant:|\Z)", conv, re.DOTALL)
    return turns[-1].strip() if turns else conv.strip()


def last_assistant_turn(conv):
    turns = re.findall(r"\n\nAssistant: (.*?)(?=\n\nHuman:|\Z)", conv, re.DOTALL)
    return turns[-1].strip() if turns else ""


hh = load_dataset("Anthropic/hh-rlhf")

rows = []
for ex in hh["train"]:
    prompt = last_human_turn(ex["chosen"])
    chosen = last_assistant_turn(ex["chosen"])
    rejected = last_assistant_turn(ex["rejected"])
    if prompt and chosen and rejected:
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

pd.DataFrame(rows).to_csv("train.csv", index=False)
```

### KTO

Expand each preference pair into two rows with a boolean `label`:

```python
kto_rows = []
for _, row in df.iterrows():
    kto_rows.append({"prompt": row["prompt"], "response": row["chosen"],   "label": True})
    kto_rows.append({"prompt": row["prompt"], "response": row["rejected"], "label": False})

pd.DataFrame(kto_rows).to_csv("train_kto.csv", index=False)
```

## Training

### DPO

=== "CLI"

    ```bash
    ludwig train --config examples/alignment/config_dpo.yaml --dataset data/train.csv
    ```

=== "Python"

    ```python
    import logging
    import yaml
    from ludwig.api import LudwigModel

    config = yaml.safe_load("""
    model_type: llm
    base_model: meta-llama/Llama-3.1-8B

    adapter:
      type: lora
      r: 16
      alpha: 32
      dropout: 0.05

    trainer:
      type: dpo
      epochs: 1
      learning_rate: 5.0e-7
      batch_size: 2
      gradient_accumulation_steps: 8
      beta: 0.1

    input_features:
      - name: prompt
        type: text

    output_features:
      - name: chosen
        type: text

    backend:
      type: local
    """)

    model = LudwigModel(config=config, logging_level=logging.INFO)
    train_stats, _, output_dir = model.train(dataset="data/train.csv")
    ```

### KTO

=== "CLI"

    ```bash
    ludwig train --config examples/alignment/config_kto.yaml --dataset data/train_kto.csv
    ```

=== "Python"

    Change the trainer block and output feature name:

    ```python
    config["trainer"] = {
        "type": "kto",
        "epochs": 1,
        "learning_rate": 5e-7,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "beta": 0.1,
        "desirable_weight": 1.0,
        "undesirable_weight": 1.0,
    }
    config["output_features"] = [{"name": "response", "type": "text"}]
    ```

### ORPO

=== "CLI"

    ```bash
    ludwig train --config examples/alignment/config_orpo.yaml --dataset data/train.csv
    ```

ORPO uses the same `prompt`/`chosen`/`rejected` columns as DPO. The trainer merges the SFT
and alignment losses so no reference model is needed — this saves roughly half the GPU memory
compared to DPO.

## Upload to HuggingFace Hub

After training, share the aligned model:

=== "CLI"

    ```bash
    ludwig upload hf_hub -r <your_org>/<model_name> -m results/experiment_run/model
    ```

=== "Python"

    ```python
    from ludwig.api import LudwigModel

    LudwigModel.upload_to_hf_hub("your_org/model_name", "results/experiment_run/model")
    ```

## See also

- [Fine-tuning guide](../../user_guide/llms/finetuning.md) — standard SFT, LoRA, and quantization options
- [LLM configuration reference](../../configuration/large_language_model.md) — full config schema
- [Llama-2 fine-tuning example](https://github.com/ludwig-ai/ludwig/tree/main/examples/llama2_7b_finetuning_4bit) — QLoRA instruction tuning
