In Ludwig, users have the option to set `batch_size` to a fixed value as part of the training config.
```
trainer:
  batch_size: 128
```
If the batch size is unspecified Ludwig sets `batch_size=auto`.

```
trainer:
  batch_size: auto
```
`auto` enables Ludwig to select an efficient batch size automatically. The actual value of the batch size can be found in training logs and in the model output directory.

Batch size tuning is supported in single-node and multi-node CPU and GPU settings.

## ECD Models

Batch size tuning for ECD models follows this procedure, starting from batch size 1:
1. Perform a small number of forward passes through the model using a sample from the dataset
2. If the model does not hit a memory error, increment the batch size and repeat from step 1. Otherwise, use the last valid batch size.

## LLMs
The main element that separates LLM batch size tuning from its ECD counterpart is the sequence length. LLM's thus undergo the batch size tuning process as ECD models with the exception being that, instead of using a random sample from the dataset, the forward passes use a synthetic data sample with a sequence length equal to the longest sequence length in the provided dataset.