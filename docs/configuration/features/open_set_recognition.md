# Open-Set Recognition

Standard classifiers are trained to output high-confidence predictions for *every* input — even
inputs from classes never seen during training.  This is called **network agnostophobia**: the
network is incapable of saying "I don't know."

Ludwig implements two loss functions from Dhamija et al., NeurIPS 2018
([paper](https://arxiv.org/abs/1811.04110)) that reduce this problem:

| Loss | Use on | Behaviour |
|------|--------|-----------|
| `entropic_open_set` | `category`, `binary` | CE on known samples + entropy maximisation on background samples |
| `objectosphere` | `category`, `binary` | CE + norm push (known) + entropy + norm suppression (background) |

Both losses require a **background class** to be present in the training data: a special label
assigned to the unknown/unwanted inputs that the model should be uncertain about.

---

## How they work

### Entropic Open-Set Loss

For **known-class** samples the loss is standard cross-entropy.
For **background** samples the loss maximises Shannon entropy of the output distribution,
driving the softmax probabilities toward uniform.

$$
\mathcal{L} = \underbrace{-\log p_y}_{\text{CE on known}} \;+\; \underbrace{\sum_k p_k \log p_k}_{\text{neg-entropy on background}}
$$

### Objectosphere Loss

Extends the entropic loss with a magnitude objective that creates a clear *norm threshold* for
unknown detection at inference time:

- **Known samples**: CE + hinge `max(0, ξ – ||z||)²` pushes logit norms ≥ ξ
- **Background samples**: entropy maximisation + `ζ ||z||²` suppresses logit norms toward zero

$$
\mathcal{L} =
  \underbrace{\text{CE}(z_{\text{known}}) + \max(0,\,\xi - \|z\|)^2}_{\text{known}} \;+\;
  \underbrace{\sum_k p_k \log p_k + \zeta\,\|z\|^2}_{\text{background}}
$$

After training, a sample with `||logits|| < threshold` can be flagged as unknown without any
extra model head.

---

## Configuration

### Entropic Open-Set Loss

```yaml
output_features:
  - name: label
    type: category
    loss:
      type: entropic_open_set
      background_class: 4   # integer index of the background/unknown class
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `background_class` | `int \| null` | `null` | Index of the background class in the feature vocabulary. When `null` the loss reduces to standard CE. |
| `weight` | `float` | `1.0` | Overall loss weight |

### Objectosphere Loss

```yaml
output_features:
  - name: label
    type: category
    loss:
      type: objectosphere
      background_class: 4
      xi: 10.0
      zeta: 0.1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `background_class` | `int \| null` | `null` | Index of the background class. When `null`, applies CE + hinge to all samples. |
| `xi` | `float` | `10.0` | Minimum logit L2 norm for known-class samples |
| `zeta` | `float` | `0.1` | Weight for the unknown-class magnitude penalty |
| `weight` | `float` | `1.0` | Overall loss weight |

---

## Finding the background class index

Ludwig assigns integer indices to category values based on their frequency in the training data
(most frequent first).  To find the index of your background class:

1. Run a training job without the agnostophobia loss (use `softmax_cross_entropy`).
2. Open `results/<experiment>/model/training_set_metadata.json`.
3. Find the output feature's `str2idx` dictionary.
4. Use the integer value next to your background class string.

Alternatively, pass the background class as the *most frequent* category in your training CSV —
that will always receive index 0.

---

## Dataset setup

The training set must contain both known-class samples and background samples.  Background samples
can come from:

- A dedicated "unknown" category in the original data.
- Out-of-distribution images or text from a separate source.
- Synthetic noise or augmented samples unlikely to appear at deployment time.

```
label,feature_1,feature_2
cat,0.3,0.7
dog,0.8,0.1
...
background,0.5,0.5   ← background/unknown samples
background,0.2,0.9
```

---

## Example: synthetic Gaussian dataset

The following reproduces the key experiment from the paper: four known Gaussian clusters
(classes 0–3) plus two unknown clusters (background class 4).

```python
# Run the standalone example (no data download required):
cd examples/open_set_recognition
python train_open_set.py
```

Expected output:

```
Model                  | Max-prob (known) | Max-prob (unknown) | Norm known | Norm unknown
-----------------------|-----------------|-------------------|------------|-------------
CE Baseline            |           0.998  |              0.741 |      8.828 |        5.375
Entropic Open-Set      |           0.974  |              0.273 |      6.254 |        0.637
Objectosphere          |           0.874  |              0.363 |     13.843 |        2.361
```

The agnostophobia losses clearly reduce the model's confidence on unknown inputs while
preserving accuracy on known classes.  Objectosphere additionally creates a large norm
gap between known (≈ 5.2) and unknown (≈ 0.2) samples, enabling near-perfect unknown
detection with a simple threshold.

---

## Inference-time unknown detection

### Using max softmax probability (both losses)

```python
from ludwig.api import LudwigModel
import torch

model = LudwigModel.load("results/my_experiment/model")
preds, _, _ = model.predict(dataset=df)

# Column added by Ludwig's postprocessor
max_probs = preds["label_probability"]
is_unknown = max_probs < 0.5  # tune threshold on validation set
```

### Using logit norm (Objectosphere only)

```python
# Collect logits from the model before the softmax
acts = model.collect_activations(layer_names=["label/decoder/logits"], dataset=df)
logit_norms = acts["label/decoder/logits"].norm(dim=-1)
is_unknown = logit_norms < 3.0  # tune threshold on validation set
```

---

## References

Dhamija, A. R., Günther, M., & Boult, T. (2018).
*Reducing Network Agnostophobia.* NeurIPS 2018.
https://arxiv.org/abs/1811.04110
