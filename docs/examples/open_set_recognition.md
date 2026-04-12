# Open-Set Recognition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/open_set_recognition/open_set_mnist.ipynb)

Standard image classifiers assign a class label — and high confidence — to *every* input, including
images from categories they have never seen during training.  This is called **network
agnostophobia**: the model is incapable of expressing "I don't know."

This example trains three Ludwig image classifiers on MNIST digits and measures how well each one
detects the unseen classes at test time.

| Model | Loss | Behaviour |
|-------|------|-----------|
| CE Baseline | `softmax_cross_entropy` | Confidently misclassifies unknown digits |
| Entropic Open-Set | `entropic_open_set` | Pushes unknown confidence toward uniform |
| Objectosphere | `objectosphere` | Creates a logit-norm gap for norm-based detection |

**Paper**: Dhamija, Günther, Boult — *Reducing Network Agnostophobia*, NeurIPS 2018.
<https://arxiv.org/abs/1811.04110>

For full configuration reference see
[Open-Set Recognition losses](../../configuration/features/open_set_recognition.md).

---

## Dataset preparation

We use MNIST with a simple known/unknown split:

- **Known classes** — digits 0–7, used for training and evaluation
- **Unknown classes** — digits 8 and 9, relabelled as `"background"` in training, kept with their
  true labels in the test set

Images are saved as PNG files; two CSVs (`train.csv`, `test.csv`) with `image_path` and `label`
columns are passed to Ludwig.

```python
import csv
from pathlib import Path
from collections import defaultdict

import torch
from torchvision import datasets, transforms
from PIL import Image

DATA_DIR = Path("mnist_data")
IMG_DIR  = Path("mnist_images")
KNOWN_CLASSES   = list(range(8))
UNKNOWN_CLASSES = [8, 9]

mnist_train = datasets.MNIST(str(DATA_DIR), train=True,  download=True,
                              transform=transforms.ToTensor())
mnist_test  = datasets.MNIST(str(DATA_DIR), train=False, download=True,
                              transform=transforms.ToTensor())

def save_image(tensor, split, digit, idx):
    folder = IMG_DIR / split / str(digit)
    folder.mkdir(parents=True, exist_ok=True)
    fpath = folder / f"{idx:05d}.png"
    Image.fromarray(
        (tensor.squeeze(0).numpy() * 255).astype("uint8"), mode="L"
    ).save(fpath)
    return str(fpath)

def build_csv(dataset, csv_path, split, max_known=500, max_unknown=500,
              label_unknown_as_background=True):
    counts_known = defaultdict(int)
    counts_unknown = defaultdict(int)
    rows = []
    for global_idx, (img, digit) in enumerate(dataset):
        digit = int(digit)
        if digit in KNOWN_CLASSES:
            if counts_known[digit] >= max_known:
                continue
            path  = save_image(img, split, digit, global_idx)
            label = str(digit)
            counts_known[digit] += 1
        elif digit in UNKNOWN_CLASSES:
            if counts_unknown[digit] >= max_unknown:
                continue
            path  = save_image(img, split, digit, global_idx)
            label = "background" if label_unknown_as_background else str(digit)
            counts_unknown[digit] += 1
        else:
            continue
        rows.append({"image_path": path, "label": label})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(rows)

build_csv(mnist_train, "train.csv", "train", label_unknown_as_background=True)
build_csv(mnist_test,  "test.csv",  "test",  label_unknown_as_background=False)
```

The resulting `train.csv` has labels `"0"` through `"7"` for known digits and `"background"` for
digits 8–9.  The `test.csv` preserves the true digit labels so we can separately evaluate
performance on known vs unknown samples.

---

## Finding the background class index

Ludwig assigns integer indices to category labels sorted by frequency.  The `entropic_open_set`
and `objectosphere` losses require the integer index of the background label via the
`background_class` parameter.

After a training run, open `results/<experiment>/model/training_set_metadata.json` and look up
the `str2idx` field of the `label` output feature:

```python
import json
from pathlib import Path

with open(Path(output_dir) / "model" / "training_set_metadata.json") as f:
    metadata = json.load(f)

BACKGROUND_CLASS = metadata["label"]["str2idx"]["background"]
print(f"background_class = {BACKGROUND_CLASS}")
```

---

## Ludwig configuration

### Baseline: softmax cross-entropy

```yaml
model_type: ecd

input_features:
  - name: image_path
    type: image
    encoder:
      type: stacked_cnn
      conv_layers:
        - num_filters: 32
          filter_size: 3
          pool_size: 2
          pool_stride: 2
        - num_filters: 64
          filter_size: 3
          pool_size: 2
          pool_stride: 2
      fc_layers:
        - output_size: 128
          dropout: 0.3

output_features:
  - name: label
    type: category
    loss:
      type: softmax_cross_entropy

trainer:
  epochs: 10
  learning_rate: 0.001
  batch_size: 128
```

The baseline is trained on known classes only (digits 0–7 with no `"background"` rows in the
training CSV).

### Entropic Open-Set loss

```yaml
output_features:
  - name: label
    type: category
    loss:
      type: entropic_open_set
      background_class: 1   # replace with value from training_set_metadata.json
```

The training CSV must include `"background"` rows.  For known samples the loss is standard
cross-entropy; for background samples it maximises Shannon entropy of the output distribution.

### Objectosphere loss

```yaml
output_features:
  - name: label
    type: category
    loss:
      type: objectosphere
      background_class: 1   # replace with value from training_set_metadata.json
      xi: 10.0              # minimum logit L2 norm for known-class samples
      zeta: 0.1             # weight for unknown-class magnitude penalty
```

The Objectosphere loss additionally pushes logit norms of known samples above `xi` and suppresses
norms for background samples toward zero, creating a clear norm threshold for unknown detection.

---

## Results

After training, the confidence histograms show a clear separation between the three approaches:

- **CE Baseline** — both known (0–7) and unknown (8–9) digits receive high max softmax probability.
  Mean max-prob on unknowns is typically ≥ 0.70.
- **Entropic Open-Set** — max-prob on unknown digits drops substantially (typically ≈ 0.25–0.35),
  approaching the uniform baseline of `1 / num_classes`.  Known-class accuracy is preserved.
- **Objectosphere** — similar max-prob reduction on unknowns, and additionally creates a large
  logit-norm gap: known-class norms cluster above `xi` while background norms cluster near zero.

The ROC AUC for unknown detection (using `1 - max_prob` as the detection score) typically improves
from ≈ 0.70 for the baseline to ≈ 0.90–0.95 for both agnostophobia models.

---

## Inference-time detection

### Max softmax probability (all models)

```python
from ludwig.api import LudwigModel

model = LudwigModel.load("results/my_experiment/model")
preds, _ = model.predict(dataset=test_df)

# Ludwig writes the winning class probability into "<feature>_probability"
max_probs = preds["label_probability"]
is_unknown = max_probs < 0.5   # tune on validation set
```

### Logit norm (Objectosphere)

```python
# Collect logits before the softmax
acts = model.collect_activations(
    layer_names=["label/decoder/logits"], dataset=test_df
)
logit_norms = acts["label/decoder/logits"].norm(dim=-1)
is_unknown = logit_norms < 3.0   # tune on validation set
```

---

## Running the example

The complete walkthrough — including data download, vocabulary discovery, training all three
models, and plotting results — is in the Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/open_set_recognition/open_set_mnist.ipynb)

Standalone YAML configs are in
[`examples/open_set_recognition/`](https://github.com/ludwig-ai/ludwig/tree/main/examples/open_set_recognition).

A self-contained Python validation script using a synthetic Gaussian dataset (no data download
required) is also available:

```bash
cd examples/open_set_recognition
pip install ludwig
python train_open_set.py
```

---

## References

Dhamija, A. R., Günther, M., & Boult, T. (2018).
*Reducing Network Agnostophobia.* NeurIPS 2018.
<https://arxiv.org/abs/1811.04110>
