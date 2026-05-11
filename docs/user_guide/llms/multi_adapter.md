---
description: Train multiple LoRA adapters on the same base model and merge them with TIES, DARE, or linear combining. Use multi-adapter PEFT for domain specialization, A/B testing, and efficient model ensembles.
---

# Multi-Adapter PEFT

Ludwig supports training and deploying **multiple named PEFT adapters** on the same base model.
This enables domain specialization, adapter ensembles, and efficient A/B testing — all without
duplicating the large base model weights.

## When to use multi-adapter PEFT

| Use case | Description |
|----------|-------------|
| **Domain specialization** | Train a "coding" adapter and a "chat" adapter separately, then merge for a model that does both |
| **A/B testing** | Switch between adapters at inference time without reloading the base model |
| **Continual learning** | Add a new adapter for each new task without forgetting previous ones |
| **Adapter ensembles** | Merge adapters trained on different data splits for better generalization |

## Configuration

Use `adapters:` (plural) instead of the singular `adapter:` to define multiple named adapters:

```yaml
model_type: llm
base_model: meta-llama/Llama-3.1-8B

adapters:
  adapters:
    coding:
      type: lora
      r: 16
      alpha: 32
      target_modules: ["q_proj", "v_proj"]
    chat:
      type: lora
      r: 8
      alpha: 16
      target_modules: ["q_proj", "v_proj"]
  active: coding  # which adapter is used at inference time
```

`adapter:` and `adapters:` are mutually exclusive — existing single-adapter configs work
unchanged.

## Merging adapters

After training, merge multiple adapters into one using PEFT's `add_weighted_adapter()`:

```yaml
adapters:
  adapters:
    coding:
      type: lora
      r: 16
    chat:
      type: lora
      r: 8
  merge:
    name: coding_chat  # name of the merged adapter
    sources: [coding, chat]
    weights: [0.7, 0.3]  # relative weights; will be normalised
    combination_type: ties
    density: 0.7         # fraction of deltas to retain (for TIES/DARE)
  active: coding_chat
```

### Combination strategies

| Strategy | Paper | Description |
|----------|-------|-------------|
| `linear` | — | Weighted sum of LoRA weight deltas. Fast, no pruning. |
| `svd` | — | SVD-based merge; reduces rank after combining |
| `ties` | [Yadav et al., NeurIPS 2023](https://arxiv.org/abs/2306.01708) | Resolves sign conflicts between adapters before summing deltas |
| `dare_linear` | [Yu et al., ICML 2024](https://arxiv.org/abs/2402.03432) | Prunes fraction `(1-density)` of deltas, then linear merge |
| `dare_ties` | [Yu et al., ICML 2024](https://arxiv.org/abs/2402.03432) | Prunes deltas, then TIES sign-conflict resolution |
| `magnitude_prune` | — | Keeps top-magnitude deltas before merging |

**Picking a strategy:**

- Start with `linear` — it's fastest and often competitive
- Use `ties` when adapters were trained on conflicting tasks (coding vs chat, different languages)
- Use `dare_linear` or `dare_ties` when you want sparse, memory-efficient merged adapters
- `density: 0.7` means keep 70% of deltas; lower values produce sparser adapters

## Training multiple adapters

Train adapters separately and merge at the end:

```python
from ludwig.api import LudwigModel
import pandas as pd

# Train coding adapter
config_coding = {
    "model_type": "llm",
    "base_model": "meta-llama/Llama-3.1-8B",
    "adapters": {"adapters": {"coding": {"type": "lora", "r": 16}}},
    "input_features": [{"name": "prompt", "type": "text"}],
    "output_features": [{"name": "response", "type": "text"}],
    "trainer": {"type": "finetune", "epochs": 3},
}
model = LudwigModel(config=config_coding)
model.train(dataset="coding_dataset.csv", output_directory="./coding_model")

# Train chat adapter
config_chat = {**config_coding}
config_chat["adapters"]["adapters"] = {"chat": {"type": "lora", "r": 8}}
model2 = LudwigModel(config=config_chat)
model2.train(dataset="chat_dataset.csv", output_directory="./chat_model")
```

## Switching adapters at inference time

With multiple adapters loaded, switch between them without reloading the model:

```python
from ludwig.api import LudwigModel

model = LudwigModel.load("results/multi_adapter_model")

# Access the underlying PEFT model
peft_model = model.model.model

# Switch active adapter
peft_model.set_adapter("coding")
coding_preds, _ = model.predict(dataset=coding_test_df)

peft_model.set_adapter("chat")
chat_preds, _ = model.predict(dataset=chat_test_df)
```

## Memory considerations

Multi-adapter PEFT loads all adapter weights simultaneously but shares the base model:

| Setup | Memory overhead vs single adapter |
|-------|----------------------------------|
| 2 × LoRA-r8 adapters | ~2× adapter memory (small vs base model) |
| Merged adapter | No overhead — same as single adapter |
| Enabled adapter switching | All adapter weights in VRAM |

For large numbers of adapters, consider training sequentially and using `merge:` to combine
them — the merged adapter has the same memory footprint as a single adapter.

## See also

- [LLM Fine-Tuning](finetuning.md) — single-adapter PEFT with LoRA/QLoRA
- [LLM configuration reference](../../configuration/large_language_model.md#multi-adapter-peft) — full YAML schema
- [Multi-adapter example](../../examples/multi_adapter.md) — runnable walkthrough
