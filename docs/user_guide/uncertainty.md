# Uncertainty Quantification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/uncertainty/uncertainty.ipynb)

Deep learning classifiers are often **overconfident**: they assign extreme predicted probabilities (near 0 or 1) even for borderline examples, and they cannot reliably signal when they "don't know" the answer. This is a real problem in any application where you use probabilities for risk scoring, decision thresholds, or routing samples to human reviewers.

Ludwig provides two built-in mechanisms for addressing model uncertainty:

- **Temperature Scaling**: post-hoc calibration that makes probabilities more faithful to empirical frequencies
- **MC Dropout**: stochastic inference that produces per-sample uncertainty estimates

Both are configured on the output feature decoder and require no changes to model architecture or training procedure.

---

## Temperature Scaling Calibration

### What it is

Temperature scaling (Guo et al., ICML 2017) is the simplest and most reliable post-hoc calibration method. After training, it learns a single scalar **T** by minimising negative log-likelihood on the validation set. At inference time, logits are divided by T before the final sigmoid/softmax:

```
calibrated_logit = logit / T
```

- **T > 1**: softens the distribution, reducing overconfidence
- **T < 1**: sharpens the distribution
- **T = 1**: no change

Temperature scaling **never changes argmax predictions** — accuracy is identical to the uncalibrated model. It only adjusts the probability values.

### When to use it

Use temperature scaling when:

- You need well-calibrated probability outputs for downstream decisions (thresholding, ranking, risk scoring)
- Your model is systematically overconfident (common with large neural networks and class-imbalanced datasets)
- You have a held-out validation set available for calibration
- Inference latency cannot increase (temperature scaling adds zero overhead)

### Configuration

Temperature scaling is enabled by setting `calibration: temperature_scaling` in the decoder configuration. It is currently supported by the `classifier` and `mlp_classifier` decoders (category and binary output features).

```yaml
output_features:
  - name: label
    type: binary
    decoder:
      type: mlp_classifier
      calibration: temperature_scaling  # learns T on the validation set after training
```

For category outputs:

```yaml
output_features:
  - name: cover_type
    type: category
    decoder:
      type: classifier
      calibration: temperature_scaling
```

### Evaluating calibration

The standard calibration metric is **Expected Calibration Error (ECE)** — the weighted average absolute gap between predicted confidence and empirical accuracy across confidence bins. Lower is better; a perfectly calibrated model has ECE = 0.

A **reliability diagram** plots predicted probability (x-axis) against empirical accuracy (y-axis). A perfectly calibrated model lies on the diagonal.

```python
import numpy as np

def expected_calibration_error(probabilities, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probabilities)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= lo) & (probabilities < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(probabilities[mask].mean() - labels[mask].mean())
    return ece
```

### Full example

```python
import copy
from ludwig.api import LudwigModel

config = {
    "model_type": "ecd",
    "input_features": [...],
    "output_features": [
        {
            "name": "label",
            "type": "binary",
            "decoder": {"type": "mlp_classifier"},
        }
    ],
    "trainer": {"epochs": 30},
}

# Baseline — no calibration
baseline = LudwigModel(config=config)
baseline.train(dataset=df)

# Temperature scaling — one line change
calibrated_config = copy.deepcopy(config)
calibrated_config["output_features"][0]["decoder"]["calibration"] = "temperature_scaling"

calibrated = LudwigModel(config=calibrated_config)
calibrated.train(dataset=df)

# Accuracy is the same; ECE should be lower
_, baseline_preds, _ = baseline.predict(dataset=df)
_, calibrated_preds, _ = calibrated.predict(dataset=df)
```

---

## MC Dropout

### What it is

MC Dropout (Gal & Ghahramani, ICML 2016) reinterprets standard dropout as approximate Bayesian inference. At inference time, instead of disabling dropout (the standard behaviour), it performs **T stochastic forward passes** with dropout enabled. The final prediction is the mean over these passes, and the **variance** is reported as an uncertainty estimate:

```
p̄  = (1/T) Σ sigmoid(f_θ^(t)(x))     # mean prediction
σ² = Var_t[sigmoid(f_θ^(t)(x))]       # per-sample uncertainty
```

High variance means the model gives inconsistent answers across passes — a signal that it genuinely does not know the answer for that sample.

### When to use it

Use MC Dropout when:

- You need per-sample uncertainty estimates, not just better-calibrated probabilities
- You want to flag low-confidence predictions for human review
- You are doing active learning (prioritise labelling high-uncertainty samples)
- You are building safety-critical systems where "I don't know" is a valid output
- You are prepared to accept ~T× slower inference (T = mc_dropout_samples)

### Configuration

MC Dropout is enabled by setting `mc_dropout_samples` to a value greater than 0 on a `classifier` or `mlp_classifier` decoder. You must also ensure `dropout > 0` in the decoder (and typically in the combiner) for stochastic variation to occur.

```yaml
output_features:
  - name: label
    type: binary
    decoder:
      type: mlp_classifier
      dropout: 0.3              # must be > 0 for variance across MC passes
      mc_dropout_samples: 20    # number of stochastic forward passes at inference

combiner:
  type: concat
  dropout: 0.2                  # combiner dropout also contributes to variance
```

### Output columns

When `mc_dropout_samples > 0`, Ludwig adds an `uncertainty` column to prediction outputs alongside the standard `predictions` and `probabilities` columns:

| Column | Description |
|---|---|
| `{feature}_predictions` | Argmax of mean probability across MC passes |
| `{feature}_probability_True` | Mean predicted probability across passes |
| `{feature}_uncertainty` | Variance of probability across passes |

### Practical tips

- A good starting value is `mc_dropout_samples: 20`. More passes improve the uncertainty estimate but increase latency linearly.
- If uncertainty values are all near zero, the dropout rate is too low or the model has insufficient capacity in the stochastic layers.
- Uncertainty is highest near the decision boundary (predicted probability ≈ 0.5), which is expected. High uncertainty near 0 or 1 is more informative — it signals a confident-but-inconsistent model.
- The uncertainty estimate is in units of probability variance (range [0, 0.25] for binary outputs).

### Full example

```python
from ludwig.api import LudwigModel
import numpy as np

config = {
    "model_type": "ecd",
    "input_features": [...],
    "output_features": [
        {
            "name": "label",
            "type": "binary",
            "decoder": {
                "type": "mlp_classifier",
                "dropout": 0.3,
                "mc_dropout_samples": 20,
            },
        }
    ],
    "combiner": {"type": "concat", "dropout": 0.2},
    "trainer": {"epochs": 30},
}

model = LudwigModel(config=config)
model.train(dataset=train_df)

_, predictions, _ = model.predict(dataset=test_df)

# Predictions include an uncertainty column
uncertainty = predictions["label_uncertainty"].values
mean_prob = predictions["label_probability_True"].values

# Flag high-uncertainty samples for review
threshold = np.percentile(uncertainty, 80)
flagged = test_df[uncertainty >= threshold]
print(f"Flagged {len(flagged)} samples for human review")
```

---

## Choosing Between Them

| | Temperature Scaling | MC Dropout |
|---|---|---|
| **Goal** | Better-calibrated probabilities | Per-sample uncertainty estimates |
| **Changes accuracy?** | No | Minimally (mean over T passes) |
| **Inference overhead** | None | ~T× slower |
| **Requires validation set?** | Yes | No |
| **Requires dropout > 0?** | No | Yes |
| **Config key** | `decoder.calibration: temperature_scaling` | `decoder.mc_dropout_samples: N` |
| **Extra output columns** | None (probabilities are rescaled) | `{feature}_uncertainty` |
| **Best for** | Probability reliability, decision thresholds, ranking | Active learning, human-in-the-loop, safety review |

### Can you use both?

Yes. Temperature scaling and MC Dropout are orthogonal and can be combined on the same decoder:

```yaml
decoder:
  type: mlp_classifier
  dropout: 0.3
  calibration: temperature_scaling  # calibrated mean probability
  mc_dropout_samples: 20            # + per-sample uncertainty estimate
```

This gives you well-calibrated probabilities (useful for scoring and thresholds) **and** per-sample uncertainty estimates (useful for routing decisions), at the cost of ~20× slower inference.

---

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). [On calibration of modern neural networks](https://arxiv.org/abs/1706.04599). *ICML*.
- Gal, Y., & Ghahramani, Z. (2016). [Dropout as a Bayesian approximation: Representing model uncertainty in deep learning](https://arxiv.org/abs/1506.02142). *ICML*.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.
