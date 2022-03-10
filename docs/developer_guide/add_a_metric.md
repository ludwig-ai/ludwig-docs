Metrics are used to report model performance during training and evaluation, and also serve as optimization objectives
for [hyperparameter optimization](../hyper_parameter_optimization).

Concretely, metrics are modules which compute a function of the model's output for each batch and aggregate the
function's result over all batches. A common example of a metric is the `LossMetric`, which computes the average batch
loss. Metrics are defined in `ludwig/modules/metric_modules.py`. Ludwig's metrics are designed to be consistent with
`torchmetrics` and conform to the interface of `torchmetrics.Metric`.

!!! note

    Before implementing a new metric from scratch, check the
    [torchmetrics documentation](https://torchmetrics.readthedocs.io/en/latest/) to see if the desired function is
    available there. Torch metrics can often be added to Ludwig trivially, see `RMSEMetric` in
    `ludwig/modules/metric_modules.py` for example.

# 1. Add a new metric class

For the majority of use cases metrics should be averaged over batches, for this Ludwig provides a `MeanMetric` class
which keeps a running average of its values. The following examples will assume averaging is desired and inherit from
`MeanMetric`. If you need different aggregation behavior replace `MeanMetric` with `LudwigMetric` and accumulate the
metric values as needed.

We'll use `TokenAccuracyMetric` as an example, which treats each token of a sequence as an independent prediction and
computes average accuracy over sequences.

First, declare the new metric class in `ludwig/modules/metric_modules.py`:
```python
class TokenAccuracyMetric(MeanMetric):
```

# 2. Implement required methods

## get_current_value

If using `MeanMetric`, compute the value of the metric given a batch of feature outputs and target values in
`get_current_value`.

```python
def get_current_value(
        self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Compute metric over a batch of predictions (preds) and truth values (target).
    # Aggregate metric over batch.
    return metric_value
```

__Inputs__

- __preds__ (torch.Tensor): A batch of outputs from an output feature which are either predictions, probabilities, or
logits depending on the return value of [get_inputs](#get_inputs).
- __target__ (torch.Tensor): The batch of true labels for the dataset column corresponding to the metric's output
feature.

__Return__

- (torch.Tensor): The computed metric, in most cases this will be a scalar value.

## update and reset

If not using `MeanMetric`, implement `update` and `reset` instead of `get_current_value`.

```python
def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
    # Compute metric over a batch of predictions (preds) and truth values (target).
    # Accumulate metric values or aggregate statistics.
```

__Inputs__

- __preds__ (torch.Tensor): A batch of outputs from an output feature which are either predictions, probabilities, or
logits depending on the return value of [get_inputs](#get_inputs).
- __target__ (torch.Tensor): The batch of true labels for the dataset column corresponding to the metric's output
feature.

```python
def reset(self) -> None:
    # Reset accumulated values.
```

!!! note
    `MeanMetric`'s update method simply delegates metric computation to `get_current_value`.
    ```python
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.avg.update(self.get_current_value(preds, target))
    ```

## get_objective

The return value of `get_objective` tells Ludwig whether to minimize or maximize this metric in hyperparameter
optimization.

```python
@classmethod
def get_objective(cls):
    return MAXIMIZE
```

__Return__

- (str): How this metric should be optimized, one of MINIMIZE or MAXIMIZE.

## get_inputs

Determines which feature output is passed in to this metric's `update` or `get_current_value` method. Valid return
values are:

- `PREDICTIONS`: The predicted values of the output feature.
- `PROBABILITIES`: The vector of probabilities.
- `LOGITS`: The vector of outputs of the feature decoder's final layer (before the application of any sigmoid or softmax
function).

```python
@classmethod
def get_inputs(cls):
    return PREDICTIONS
```

__Return__

- (str): Which output this metric derives its value from, one of `PREDICTIONS`, `PROBABILITIES`, or `LOGITS`.

# 3. Add the new metric class to the registry

Mapping between metric names in the config and metric classes is made by registering the class in a metric registry. The
metric registry is defined in `ludwig/modules/metric_registry.py`. To register your class, add the `@register_metric`
decorator on the line above its class definition, specifying the name of the metric and a list of the supported output
feature types:

```python
@register_metric(TOKEN_ACCURACY, [SEQUENCE, TEXT])
class TokenAccuracyMetric(MeanMetric):
```
g