# Advanced PEFT Adapters

Ludwig's PEFT integration (backed by HuggingFace PEFT) goes well beyond standard LoRA. PR #4146 adds
new LoRA initializers that improve convergence and final quality, per-module rank/alpha overrides, and
several new adapter families covering orthogonal, wavelet-based, and layer-norm-only tuning strategies.

This page collects config snippets and short explanations for each advanced option. For the full
parameter reference see the [LLM configuration docs](../../configuration/large_language_model.md#adapter).

## LoRA initializers

Standard LoRA initializes `B = 0` so the adapter is a no-op at the start of training. The initializers
below start from a better point, which speeds up convergence and often improves the final metric.

### PiSSA

**Principal Singular Values and Singular Vectors Adaptation** aligns the low-rank subspace with the
top-`r` singular components of each pretrained weight matrix. The residual is kept frozen. PiSSA
consistently outperforms standard LoRA at the same rank and requires no extra data.

```yaml
model_type: llm
base_model: meta-llama/Llama-3.1-8B

adapter:
  type: lora
  r: 16
  alpha: 16
  init_lora_weights: pissa

trainer:
  type: finetune
  epochs: 3
  learning_rate: 1e-4
```

### CorDA

**Correlation-Driven LoRA Adaptation** initializes the subspace from activation correlations computed on
a small calibration batch. It is most effective when a representative in-domain sample is available at
initialization time.

```yaml
adapter:
  type: lora
  r: 16
  alpha: 16
  init_lora_weights: corda
```

## Layer-Norm tuning

Layer-Norm tuning trains only the `weight` and `bias` parameters of LayerNorm/RMSNorm layers. It is the
lightest adapter type available — often fewer than 0.1% of backbone parameters — and works well for
domain adaptation of already-instruction-tuned models where the knowledge is largely intact and only the
output distribution needs to shift.

```yaml
model_type: llm
base_model: mistralai/Mistral-7B-Instruct-v0.3

input_features:
  - name: prompt
    type: text
output_features:
  - name: response
    type: text

adapter:
  type: ln_tuning

trainer:
  type: finetune
  epochs: 2
  learning_rate: 5e-4
  batch_size: 4
  gradient_accumulation_steps: 8
```

## Orthogonal adapters

### OFT

**Orthogonal Fine-Tuning** constrains weight updates to orthogonal transformations, preserving the
hyperspherical geometry of the pretrained representations. This keeps the relative angles between
token embeddings stable during fine-tuning and is particularly effective for tasks that depend on
semantic similarity structure.

```yaml
adapter:
  type: oft
  r: 8
  module_dropout: 0.0
```

### HRA

**Householder Reflection Adaptation** parameterizes updates as a product of Householder reflections.
It achieves a similar orthogonality guarantee to OFT but with fewer parameters per layer.

```yaml
adapter:
  type: hra
  r: 8
```

Both OFT and HRA are drop-in replacements for LoRA in any Ludwig LLM config — just change the `type`
field on `adapter`.

## Wavelet-based tuning

### WaveFT

WaveFT applies updates in the wavelet domain, concentrating the parameter budget on the frequency
bands most perturbed during fine-tuning. It is especially useful for models that process structured
signals (audio, images encoded as tokens) where frequency structure carries semantic meaning.

```yaml
adapter:
  type: waveft
  r: 8
  alpha: 16
```

## Vector-bank adapters

### VBLoRA

**Vector-Bank LoRA** replaces the per-layer `B` matrix with a shared global dictionary of vectors. Each
layer selects a subset of these vectors and linearly combines them. When many layers learn similar update
directions this can reduce total parameter count significantly versus standard LoRA.

```yaml
adapter:
  type: vblora
  r: 4
  num_vectors: 256
  vector_length: 256
```

## Comparison table

| Adapter | Extra params (approx.) | Key strength | Best for |
|---------|------------------------|--------------|----------|
| `lora` (default init) | ~0.1–1% | Versatile, well-studied | General fine-tuning baseline |
| `lora` + `pissa` | ~0.1–1% | Better initialization, faster convergence | When standard LoRA underfits |
| `lora` + `corda` | ~0.1–1% | Data-driven subspace alignment | In-domain adaptation with calibration data |
| `lora` + `loftq` | ~0.1–1% | Minimises quantization error at init | 4-bit QLoRA fine-tuning |
| `ln_tuning` | <0.1% | Extremely lightweight | Domain shift with instruction-tuned base models |
| `oft` | ~0.5–2% | Preserves hyperspherical geometry | Semantic similarity, generation fidelity |
| `hra` | ~0.3–1% | Orthogonal, fewer params than OFT | Same as OFT with tighter parameter budget |
| `waveft` | ~0.5–2% | Frequency-domain concentration | Audio/vision token models |
| `vblora` | ~0.05–0.5% | Shared vector bank across layers | Very low parameter budgets |
| `c3a` | ~0.2–1% | Block-sparse updates | Sparse activation models |
| `tinylora` | ~0.05–0.5% | Learned rank allocation | Strict parameter count constraints |
