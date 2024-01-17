To maximize efficiency, Ludwig performs automatic batch size tuning when the `batch_size` parameter is noet in the configuration in order to best saturate the GPU. Batch size tuning does not occur during CPU training due to the lack of effective parallelization, and Ludwig instead sets the batch size to a fixed value.

## ECD Models

Batch size tuning for ECD models follows this procedure, starting from batch size 1:
1. Perform a small number of forward passes through the model using a sample from the dataset
2. If the model does not hit a memory error, increment the batch size and repeat from step 1. Otherwise, use the last valid batch size.

## LLMs
The main element that separates LLM batch size tuning from its ECD counterpart is the sequence length. LLM's thus undergo the batch size tuning process as ECD models with the exception being that, instead of using a random sample from the dataset, the forward passes use a synthetic data sample with a sequence length equal to the longest sequence length in the provided dataset.