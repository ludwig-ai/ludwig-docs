---
description: "K-fold cross-validation with Ludwig — run repeated stratified splits for reliable model evaluation."
---

# K-Fold Cross-Validation

K-fold cross-validation gives a more reliable performance estimate than a single train/val/test split,
especially for small datasets.  Ludwig supports k-fold CV through the Python API.

## How it works

1. The dataset is split into *k* non-overlapping folds.
2. For each fold *i*, the model is trained on the remaining *k − 1* folds and evaluated on fold *i*.
3. Final metrics are averaged across all *k* folds.

## Example: binary classification on Titanic

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ludwig.api import LudwigModel
import numpy as np

config = {
    "model_type": "ecd",
    "input_features": [
        {"name": "Pclass", "type": "category"},
        {"name": "Sex", "type": "category"},
        {"name": "Age", "type": "number", "preprocessing": {"missing_value_strategy": "fill_with_mean"}},
        {"name": "SibSp", "type": "number"},
        {"name": "Fare", "type": "number"},
    ],
    "output_features": [
        {"name": "Survived", "type": "binary"}
    ],
    "trainer": {"epochs": 20},
}

df = pd.read_csv("titanic.csv")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["Survived"])):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    model = LudwigModel(config=config, logging_level=0)
    model.train(
        training_set=train_df,
        validation_set=val_df,
        output_directory=f"results/fold_{fold}",
    )
    eval_stats, _, _ = model.evaluate(dataset=val_df)
    fold_metrics.append(eval_stats["Survived"]["accuracy"])
    print(f"Fold {fold+1} accuracy: {fold_metrics[-1]:.4f}")

print(f"\nMean accuracy: {np.mean(fold_metrics):.4f} ± {np.std(fold_metrics):.4f}")
```

## Stratified splitting for classification

For classification tasks with imbalanced classes, always use stratified k-fold to ensure each fold contains
approximately the same proportion of each class.  The `StratifiedKFold` example above does this automatically.

## Repeated k-fold

For very small datasets (< 200 rows), repeated k-fold further reduces variance by running k-fold multiple times
with different random seeds:

```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
# Use rskf.split() in place of skf.split() in the loop above
```

## Group k-fold

Use group k-fold when samples have a grouping structure that must not be split across folds (e.g., multiple
readings from the same patient, or multiple images from the same object):

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
groups = df["patient_id"]  # ensure no patient appears in both train and val

for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["label"], groups)):
    ...
```

## Time-series split

For time series data, never shuffle — use a rolling-window split to preserve temporal order:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
df_sorted = df.sort_values("timestamp")

for fold, (train_idx, val_idx) in enumerate(tscv.split(df_sorted)):
    ...
```

## Aggregating results

```python
import numpy as np

# After collecting fold_metrics list:
mean = np.mean(fold_metrics)
std = np.std(fold_metrics)
ci_95 = 1.96 * std / np.sqrt(len(fold_metrics))  # 95% confidence interval

print(f"Accuracy: {mean:.4f} ± {ci_95:.4f} (95% CI)")
```

## Distributed k-fold with Ray

Running each fold in parallel on a Ray cluster reduces wall-clock time linearly with the number of available workers:

```python
import ray
from ludwig.backend import RAY

ray.init()

config["backend"] = {"type": "ray", "trainer": {"num_workers": 4}}
# Then run the same loop — Ludwig will distribute each fold across 4 GPU workers
```

## When to use k-fold

| Dataset size | Recommendation |
|-------------|----------------|
| < 500 rows | 10-fold or repeated 5-fold |
| 500–5k rows | 5-fold |
| 5k–50k rows | Holdout split is usually sufficient; 5-fold if you need tight confidence intervals |
| > 50k rows | Single holdout split — k-fold provides minimal additional benefit |

For hyperparameter search, prefer a single train/val/test split (or nested k-fold) over running hyperopt inside
each k-fold, as the computational cost grows as *k × number of trials*.
