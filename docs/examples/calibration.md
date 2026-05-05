---
description: "Calibrate classification model confidence with temperature scaling in Ludwig — config, API, and evaluation examples."
---

# Probability Calibration

A well-calibrated model produces confidence scores that reflect empirical accuracy: when it says it is 80%
confident, it should be correct approximately 80% of the time.  Many neural networks are overconfident by default.

Ludwig supports **temperature scaling** (Guo et al., ICML 2017) directly in the decoder config — no
post-processing step needed.

## What is temperature scaling?

Temperature scaling divides the logits by a learnable scalar *T* before applying softmax:

```
p = softmax(logits / T)
```

- *T > 1* softens the distribution (reduces overconfidence).
- *T < 1* sharpens the distribution (rarely needed).
- *T = 1* is the uncalibrated model.

Temperature scaling never changes the **argmax** prediction — it only re-shapes the probability distribution.
The scalar *T* is fit on the validation set after normal training completes.

## Configuration

Add `calibration: temperature_scaling` to the decoder section of any category output feature:

```yaml
model_type: ecd

input_features:
  - name: text
    type: text
    encoder:
      type: bert
      use_pretrained: true
      trainable: false

output_features:
  - name: label
    type: category
    decoder:
      type: classifier
      calibration: temperature_scaling

trainer:
  epochs: 10
  optimizer:
    type: adamw
    lr: 2.0e-5
```

Ludwig automatically runs temperature scaling on the validation split at the end of training and serializes *T*
with the model.  The learned temperature is applied transparently at inference.

## Full pipeline example

```python
import pandas as pd
from ludwig.api import LudwigModel
from sklearn.model_selection import train_test_split

df = pd.read_csv("agnews.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

model = LudwigModel(config="calibrated_config.yaml", logging_level=1)
model.train(dataset=train_df, output_directory="results/")

# Predict — returned probabilities are calibrated
predictions, _ = model.predict(dataset=test_df)
print(predictions[["label_predictions", "label_probabilities_<CAT>"]])
```

## Evaluating calibration

Calibration is measured with the **Expected Calibration Error (ECE)**: the weighted average of |accuracy − confidence|
across probability buckets.  Lower is better; a perfectly calibrated model has ECE = 0.

```python
import numpy as np

def ece(confidences, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return ece

# Extract max probability from Ludwig predictions
confs = predictions["label_probabilities"].apply(max).values
labels = (predictions["label_predictions"] == test_df["label"].values).astype(float)
print(f"ECE: {ece(confs, labels):.4f}")
```

## Monte Carlo dropout uncertainty

For an alternative uncertainty estimate that does not require a validation set, use MC dropout:

```yaml
output_features:
  - name: label
    type: category
    decoder:
      type: mlp_classifier
      num_fc_layers: 2
      dropout: 0.2
      mc_dropout_samples: 20
```

With `mc_dropout_samples > 0`, Ludwig runs the decoder 20 times at inference with dropout active and returns
the **mean** prediction and an **uncertainty** tensor capturing variance across samples.

## When to calibrate

- **Safety-critical applications** where predicted confidence is acted upon (medical, finance, autonomous systems).
- **Threshold-based classifiers** where a specific confidence level triggers an action.
- **Ensembles and distillation** — calibrated soft targets improve student model quality.
- **Retrieval and ranking** — calibrated scores enable meaningful score comparison across classes.

Temperature scaling is lightweight and almost always improves calibration without hurting accuracy, so it is
recommended as a default for production classification models.
