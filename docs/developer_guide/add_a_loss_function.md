At a high level, a loss function evaluates how well a model predicts a dataset. Loss functions should always output a
scalar. Lower loss corresponds to a better fit, thus the objective of training is to minimize the loss.

Ludwig losses conform to the `torch.nn.Module` interface, and are declared in `ludwig/modules/loss_modules.py`. Before
implementing a new loss from scratch, check the documentation of [torch.nn loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
to see if the desired loss is available. Adding a torch loss to Ludwig is simpler than implementing a loss from scratch.

# Add a torch loss to Ludwig

Torch losses whose call signature takes model outputs and targets i.e. `loss(model(input), target)` can be added to
Ludwig easily by declaring a trivial subclass in `ludwig/modules/loss_modules.py` and registering the loss for one or
more output feature types. This example adds `MAELoss` (mean absolute error loss) to Ludwig:

```python
@register_loss("mean_absolute_error", [NUMBER, TIMESERIES, VECTOR])
class MAELoss(torch.nn.L1Loss, LogitsInputsMixin):
    def __init__(self, **kwargs):
        super().__init__()
```

The `@register_loss` decorator registers the loss under the name `mean_absolute_error`, and indicates it is supported
for `NUMBER`, `TIMESERIES`, and `VECTOR` output features.

# Implement a loss from scratch

## Implement loss function

To implement a new loss function, we recommend first implementing it as a function of logits and labels, plus any other
configuration parameters. For this example, lets suppose we have implemented the tempered softmax from
["Robust Bi-Tempered Logistic Loss Based on Bregman Divergences"](https://arxiv.org/abs/1906.03361). This loss function
takes two constant parameters `t1` and `t2`, which we'd like to allow users to specify in the config.

Assuming we have the following function:

```python
def tempered_softmax_cross_entropy_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        t1: float, t2: float) -> torch.Tensor:
    # Computes the loss, returns the result as a torch.Tensor.
```

## Define and register module

Next, we'll define a module class which computes our loss function, and add it to the loss registry for `CATEGORY`
output features with `@register_loss`. `LogitsInputsMixin` tells Ludwig that this loss should be called with the output
feature `logits`, which are the feature decoder outputs before normalization to a probability distribution.

```python
@register_loss("tempered_softmax_cross_entropy", [CATEGORY])
class TemperedSoftmaxCrossEntropy(torch.nn.Module, LogitsInputsMixin):
```

!!! note

    It is possible to define losses on other outputs besides `logits` but this is not used in Ludwig today. For
    example, loss could be computed over `probabilities`, but it is usually more numerically stable to compute from
    `logits` (rather than backpropagating loss through a softmax function).

## constructor

The loss constructor will receive any parameters specified in the config as kwargs. It must provide reasonable defaults
for all arguments.

```python
def __init__(self, t1: float = 1.0, t2: float = 1.0, **kwargs):
    super().__init__()
    self.t1 = t1
    self.t2 = t2
```

## forward

The forward method is responsible for computing the loss. Here we'll call the `tempered_softmax_cross_entropy_loss`
after ensuring its inputs are the correct type, and return its output averaged over the batch.

```python
def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    labels = target.long()
    loss = tempered_softmax_cross_entropy_loss(logits, labels, self.t1, self.t2)
    return torch.mean(loss)
```

## Define a loss schema class

In order to validate user input against the expected inputs and input types for the new loss you have defined, we need 
to create a schema class that will autogenerate the json schema required for validation. This class should be defined 
in `ludiwg.schema.features.loss.loss.py`. This example adds a schema class for the `MAELoss` class defined above:

```python
@dataclass
class MAELossConfig(BaseLossConfig):

    type: str = schema_utils.StringOptions(
        options=[MEAN_ABSOLUTE_ERROR],
        description="Type of loss.",
    )

    weight: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Weight of the loss.",
    )
```

Lastly, we need to add a reference to this schema class on the loss class. For example, on the `MAELoss` class defined 
above, we would add:

```python
    @staticmethod
    def get_schema_cls():
        return MAELossConfig
```