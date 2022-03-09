Metrics are modules which compute a function of the model's output for each batch and aggregate the function's result
over all batches. A common example of a metric is the `LossMetric`, which computes the batch loss and averages over
batches. Metrics may be used to report on model training and test performance, and also may serve as optimization
targets for [hyperparameter optimization](../hyper_parameter_optimization).


Metrics are defined in `ludwig/modules/metric_modules.py`. Ludwig metrics are designed to be consistent with
`torchmetrics`. Concretely, Ludwig metrics are subclasses of `LudwigMetric` which inherits from `torchmetrics.Metric`.

!!! note

    Before implementing a new metric from scratch, check the
    [torchmetrics documentation](https://torchmetrics.readthedocs.io/en/latest/) to see if the desired function is
    available there. Torch metrics can often be added to Ludwig trivially, for example `RMSEMetric` in
    `ludwig/modules/metric_modules.py`

# 1. Add a new metric class


# 2. Implement required methods


# 3. Add the new metric class to the registry

Mapping between metric names in the config and metric classes is made by registering the class in a metric registry. The
metric registry is defined in `ludwig/modules/metric_registry.py`. To register your class, add the `@register_metric`
decorator on the line above its class definition, specifying the name of the metric and a list of the supported output
feature types:

```python
@register_metric(ACCURACY, [CATEGORY])
class CategoryAccuracy(_Accuracy, LudwigMetric):
```
