# Loss Functions

This page documents the loss functions added in Ludwig 0.14. Each loss is registered for a
specific set of feature types and is selected by setting `loss.type` on the output feature.

For the complete loss catalog (including classic cross-entropy, MSE, MAE, and sequence
losses), see the per-feature-type pages under **Features**.

---

## Focal loss

Registered on: `binary`, `category`, `image`.

[Focal loss](https://arxiv.org/abs/1708.02002) (Lin et al., ICCV 2017) down-weights easy
examples and focuses training on hard, misclassified ones. It is the standard choice for
imbalanced classification and dense detection tasks where a small number of positives are
drowned out by the background class.

```yaml
output_features:
  - name: label
    type: category
    loss:
      type: focal_loss
      alpha: 0.25     # class-balance factor
      gamma: 2.0      # focusing parameter (0 → plain cross-entropy)
```

`gamma` controls how aggressively the loss suppresses easy examples (`(1 - p_t) ** gamma`).
`alpha` rebalances the positive / negative contributions. Use `gamma ≈ 2.0` and
`alpha ≈ 0.25` as a starting point for highly imbalanced problems.

---

## Dice loss

Registered on: `image`.

[Dice loss](https://arxiv.org/abs/1606.04797) (Milletari et al., 3DV 2016) directly
optimizes the Sørensen–Dice coefficient and is the standard loss for binary and multi-class
semantic segmentation — especially when foreground pixels are rare.

```yaml
output_features:
  - name: mask
    type: image
    decoder:
      type: unet
    loss:
      type: dice_loss
      smooth: 1.0     # Laplace smoothing added to numerator and denominator
```

Dice loss is often combined with cross-entropy in practice. To approximate that, configure
cross-entropy as the primary loss and add the Dice term as a secondary output feature with
`weight < 1.0`, or switch between the two across training stages.

---

## Lovász-Softmax loss

Registered on: `image`.

[Lovász-Softmax](https://arxiv.org/abs/1705.08790) (Berman et al., CVPR 2018) is a convex
surrogate for the mean intersection-over-union (mIoU) metric. Because it is trained
directly on the quantity that is usually used to evaluate segmentation models, it often
outperforms Dice and cross-entropy on the same task, particularly for classes with small
spatial extent.

```yaml
output_features:
  - name: mask
    type: image
    loss:
      type: lovasz_softmax_loss
```

Lovász-Softmax has no tunable hyperparameters beyond the overall `weight`. It is more
expensive than Dice because the loss is computed over sorted errors, but the extra cost is
usually negligible compared to the encoder.

---

## PolyLoss

Registered on: `category`.

[PolyLoss](https://arxiv.org/abs/2204.12511) (Leng et al., ICLR 2022) reinterprets the
cross-entropy loss as the leading term of a Taylor expansion and adds a single polynomial
correction term `epsilon * (1 - p_t)`. The result typically produces small but consistent
accuracy gains on classification benchmarks with no changes to the model or optimizer.

```yaml
output_features:
  - name: class
    type: category
    loss:
      type: poly_loss
      epsilon: 1.0
```

PolyLoss-1 (`epsilon: 1.0`) is the starting point recommended by the paper. Sweep
`epsilon ∈ [-1, 2]` if you need to tune it further.

---

## NT-Xent (contrastive)

Registered on: `vector`.

[NT-Xent](https://arxiv.org/abs/2002.05709) (normalized temperature-scaled cross-entropy,
Chen et al., ICML 2020) is the contrastive objective used by SimCLR. Ludwig applies it on
`vector` output features under the convention that consecutive rows in a batch
`(2i, 2i + 1)` form a positive pair; all other cross-pair examples in the batch act as
negatives.

```yaml
output_features:
  - name: embedding
    type: vector
    loss:
      type: nt_xent_loss
      temperature: 0.07
```

Lower temperatures produce sharper contrasts and are preferred when the embedding space is
high-dimensional (the SimCLR paper uses `0.1`). Feed the model two augmented views of each
example so that every batch contains an even number of rows ordered as pairs.

---

## Entmax-1.5

Registered on: `category`, `text`, `sequence`.

`entmax_1.5_loss` was already implemented in earlier versions but is first registered for
configuration use in 0.14. It trains the model under the sparse `entmax-1.5` projection
([Peters et al., ACL 2019](https://arxiv.org/abs/1905.05702)), which sits between softmax
and sparsemax and often yields more interpretable, sparse output distributions for
classification tasks with many labels.

```yaml
output_features:
  - name: tag
    type: category
    loss:
      type: entmax_1.5_loss
```

No extra hyperparameters are required.

---

## Open-set recognition losses

For open-set classification, see the dedicated
[Open-Set Recognition](open_set_recognition.md) page, which documents the
`entropic_open_set` and `objectosphere` losses registered on `binary` and `category`
features.

## Anomaly detection losses

For anomaly detection, see the [Anomaly Features](anomaly_features.md) page, which
documents the `deep_svdd`, `deep_sad`, and `drocc` losses registered on `anomaly` features.
