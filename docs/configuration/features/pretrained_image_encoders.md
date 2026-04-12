# Pretrained Image Encoders

Ludwig supports three HuggingFace-backed pretrained image encoders alongside the traditional `stacked_cnn` approach:

| Encoder type | Pretrained source | Default model |
|---|---|---|
| `stacked_cnn` | None (random init) | — |
| `dinov2` | Meta DINO v2 (self-supervised) | `facebook/dinov2-base` |
| `clip` | OpenAI CLIP (image-text contrastive) | `openai/clip-vit-base-patch32` |
| `siglip` | Google SigLIP (sigmoid image-text) | `google/siglip-base-patch16-224` |

All three pretrained encoders load weights from HuggingFace Hub by default and support both **linear probing** (frozen backbone) and **full fine-tuning** (trainable backbone).

## When to use pretrained vs from-scratch

Use a **pretrained encoder** when:
- Your dataset has fewer than ~10,000 labeled images
- Your images are natural photographs (similar domain to pretraining data)
- You want fast iteration and low GPU memory usage (linear probe)
- You need good accuracy on very few examples (5–50 per class)

Use **stacked_cnn** (from scratch) when:
- You have a large dataset and need maximum architectural flexibility
- Your images are highly domain-specific (medical, satellite, microscopy) where pretrained features may not transfer well
- You want full control over the convolutional architecture

## Encoders

### Convolutional Stack Encoder (`stacked_cnn`)

Stack of 2D convolutional layers, no pretrained weights. Suitable as a baseline or when training data is abundant.

```yaml
input_features:
  - name: image_path
    type: image
    encoder:
      type: stacked_cnn
      use_pretrained: false
```

### DINOv2 (`dinov2`)

Self-supervised vision transformer from Meta (Oquab et al., TMLR 2024), trained on 142M images without image-text pairs. Produces rich general-purpose features that transfer well across diverse image domains.

**Linear probe (recommended starting point):**

```yaml
input_features:
  - name: image_path
    type: image
    encoder:
      type: dinov2
      use_pretrained: true
      trainable: false   # freeze backbone, train head only
```

**Full fine-tuning:**

```yaml
input_features:
  - name: image_path
    type: image
    encoder:
      type: dinov2
      use_pretrained: true
      trainable: true
```

**Configuration parameters:**

| Parameter | Default | Description |
|---|---|---|
| `use_pretrained` | `true` | Load pretrained weights from HuggingFace Hub |
| `trainable` | `true` | Whether encoder parameters are updated during training |
| `pretrained_model_name_or_path` | `facebook/dinov2-base` | HuggingFace model identifier |

Available model variants (larger models produce higher-dimensional outputs and are more accurate but slower):

- `facebook/dinov2-small` — 22M params, 384-dim output
- `facebook/dinov2-base` — 86M params, 768-dim output (default)
- `facebook/dinov2-large` — 307M params, 1024-dim output
- `facebook/dinov2-giant` — 1.1B params, 1536-dim output

### CLIP (`clip`)

Vision transformer from OpenAI (Radford et al., ICML 2021), trained on 400M image-text pairs using contrastive learning. Produces embeddings aligned with text in a shared latent space.

Best for: image-text retrieval, zero-shot classification, multimodal fusion tasks.

```yaml
input_features:
  - name: image_path
    type: image
    encoder:
      type: clip
      use_pretrained: true
      trainable: false
```

**Configuration parameters:**

| Parameter | Default | Description |
|---|---|---|
| `use_pretrained` | `true` | Load pretrained weights from HuggingFace Hub |
| `trainable` | `true` | Whether encoder parameters are updated during training |
| `pretrained_model_name_or_path` | `openai/clip-vit-base-patch32` | HuggingFace model identifier |

Available model variants:

- `openai/clip-vit-base-patch32` — ViT-B/32, 768-dim output (default)
- `openai/clip-vit-base-patch16` — ViT-B/16, higher resolution patches
- `openai/clip-vit-large-patch14` — ViT-L/14, 1024-dim output

### SigLIP (`siglip`)

Vision transformer from Google (Zhai et al., ICCV 2023), which improves on CLIP by replacing the softmax contrastive loss with a per-image sigmoid loss. This removes dependence on global batch statistics and enables better scaling to large batch sizes and model sizes.

Best for: similar to CLIP, but often outperforms it on downstream classification. Particularly strong at small model sizes.

```yaml
input_features:
  - name: image_path
    type: image
    encoder:
      type: siglip
      use_pretrained: true
      trainable: false
```

**Configuration parameters:**

| Parameter | Default | Description |
|---|---|---|
| `use_pretrained` | `true` | Load pretrained weights from HuggingFace Hub |
| `trainable` | `true` | Whether encoder parameters are updated during training |
| `pretrained_model_name_or_path` | `google/siglip-base-patch16-224` | HuggingFace model identifier |

Available model variants:

- `google/siglip-base-patch16-224` — ViT-B/16 at 224px, 768-dim output (default)
- `google/siglip-large-patch16-256` — ViT-L/16 at 256px, 1024-dim output
- `google/siglip-so400m-patch14-384` — SO400M at 384px, 1152-dim output (highest quality)

## Linear probing vs fine-tuning

Both modes are enabled by the `trainable` parameter on the encoder config.

### Linear probing (`trainable: false`)

The pretrained backbone is completely frozen. Only the Ludwig output head (a small fully connected layer mapping encoder output to class logits) has trainable parameters.

**Advantages:**
- Very fast training (no backprop through the backbone)
- Low GPU memory (no activation storage for the backbone)
- Resistant to overfitting on small datasets
- Good accuracy even with fewer than 50 labeled examples per class

**Disadvantages:**
- Cannot adapt the backbone features to your specific domain
- Performance ceiling lower than full fine-tuning with sufficient data

**Recommended settings:** higher learning rate (0.001–0.01), more epochs (10–20)

### Full fine-tuning (`trainable: true`)

Gradients flow through the entire encoder. All parameters are updated.

**Advantages:**
- Higher accuracy ceiling, especially on domain-specific images
- Can adapt pretrained features to new domains

**Disadvantages:**
- Requires more GPU memory (stores activations for all backbone layers)
- Risk of catastrophic forgetting with too high a learning rate
- Requires more labeled data to avoid overfitting

**Recommended settings:** lower learning rate (1e-4 to 5e-5), fewer epochs (3–10), use early stopping

### Decision guide

```
Do you have fewer than 1,000 labeled images?
  YES → Use linear probe (trainable: false)
  NO  → Try fine-tuning; fall back to linear probe if overfitting

Is your domain very different from natural photographs?
  YES → Fine-tuning may be necessary to adapt features
  NO  → Linear probe likely sufficient

Do you have limited GPU memory (<8 GB)?
  YES → Use linear probe
  NO  → Either mode works
```

## Performance expectations

The following numbers are approximate, based on the `beans` dataset (~1,000 training images, 3 classes) on a T4 GPU:

| Encoder | Mode | Accuracy | Train time | Peak GPU |
|---|---|---|---|---|
| `stacked_cnn` | from scratch | ~0.65–0.75 | ~5 min | ~1 GB |
| `dinov2` | linear probe | ~0.90–0.95 | ~2 min | ~2 GB |
| `dinov2` | fine-tuned | ~0.93–0.97 | ~5 min | ~6 GB |
| `clip` | linear probe | ~0.85–0.92 | ~2 min | ~2 GB |
| `siglip` | linear probe | ~0.87–0.93 | ~2 min | ~2 GB |

!!! note
    Results vary significantly with dataset size, domain, and hyperparameters. Always run your own experiments.

## Saved weights and checkpoints

All three pretrained encoders set `saved_weights_in_checkpoint: false` by default. When you save a trained Ludwig model and reload it, Ludwig automatically sets this to `true` to load encoder weights from the checkpoint rather than re-downloading from HuggingFace. This means trained models are fully self-contained.

## Example notebook

See the [Pretrained Image Encoders example notebook](../../../examples/image_encoders/image_encoders.ipynb) for a complete walkthrough comparing all four approaches on the `beans` plant disease dataset, including a few-shot experiment with only 15 training examples.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/image_encoders/image_encoders.ipynb)
