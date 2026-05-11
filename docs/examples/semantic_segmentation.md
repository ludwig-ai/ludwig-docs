---
description: "Train a semantic segmentation model with Ludwig using a U-Net or SegFormer decoder on a pixel-level labeling task."
---

# Semantic Segmentation

Semantic segmentation assigns a class label to every pixel in an image.
Ludwig supports segmentation natively through image output features with `U-Net` and `SegFormer` decoders.

## Dataset

We use the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) where each image has a corresponding
trimap mask (foreground / background / uncertain). The CSV has two columns:

| image | mask |
|-------|------|
| images/Abyssinian_1.jpg | masks/Abyssinian_1.png |
| ... | ... |

## Config

```yaml
model_type: ecd

input_features:
  - name: image
    type: image
    encoder:
      type: convnextv2            # hierarchical CNN pretrained on ImageNet-22k
      model_name: convnextv2_tiny
      use_pretrained: true
      trainable: true

output_features:
  - name: mask
    type: image
    decoder:
      type: unet
      num_classes: 3              # foreground / background / uncertain
    loss:
      type: softmax_cross_entropy

preprocessing:
  image:
    height: 256
    width: 256
    num_channels: 3

trainer:
  epochs: 30
  batch_size: 16
  optimizer:
    type: adamw
    lr: 3.0e-4
  learning_rate_scheduler:
    type: cosine
  use_mixed_precision: true
```

## Training

```python
import ludwig
from ludwig.api import LudwigModel

model = LudwigModel(config="segmentation_config.yaml", logging_level=1)
results = model.train(dataset="pets_segmentation.csv", output_directory="results")
```

## Switching to SegFormer

[SegFormer](https://arxiv.org/abs/2105.15203) pairs a hierarchical Mix-Transformer (MiT) encoder with a
lightweight MLP decoder and achieves better accuracy than U-Net with fewer parameters.

```yaml
input_features:
  - name: image
    type: image
    encoder:
      type: vit                   # or swin for hierarchical features
      use_pretrained: true

output_features:
  - name: mask
    type: image
    decoder:
      type: segformer
      num_classes: 3
      hidden_dim: 256
```

## Feature Pyramid Network decoder

FPN decoders combine multi-scale features from the encoder backbone and produce high-resolution outputs without
requiring a symmetric up-sampling path, which reduces memory usage compared to U-Net for large images.

```yaml
output_features:
  - name: mask
    type: image
    decoder:
      type: fpn
      num_classes: 3
      num_channels: 128
      upsample_mode: bilinear
```

## Evaluation

Ludwig reports per-class pixel accuracy and mean IoU (intersection over union) automatically for segmentation tasks.

```python
eval_stats, predictions, output_dir = model.evaluate(
    dataset="pets_test.csv",
    collect_predictions=True,
    output_directory="eval_results",
)
print(eval_stats["mask"]["mean_iou"])
```

## Tips

- **Input resolution**: U-Net and SegFormer both benefit from images ≥ 256×256.  At 512×512 or larger, reduce
  `batch_size` and enable `use_mixed_precision: true`.
- **Encoder choice**: `convnextv2` and `swin` produce hierarchical features that work best with FPN/SegFormer.
  Plain `vit` works better with U-Net.
- **Class imbalance**: For highly imbalanced classes (e.g., thin boundaries), add `class_weights` to the loss.
