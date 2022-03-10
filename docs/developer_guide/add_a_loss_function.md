At a high level, a loss function evaluates how well your model predicts your data. Lower loss should correspond to a
better fit, thus the objective of training is to minimize the loss.



!!! note

    Before implementing a new metric from scratch, check the
    [torchmetrics documentation](https://torchmetrics.readthedocs.io/en/latest/) to see if the desired function is
    available there. Torch metrics can often be added to Ludwig trivially, see `RMSEMetric` in
    `ludwig/modules/metric_modules.py` for example.

# 1. Add a new loss module

