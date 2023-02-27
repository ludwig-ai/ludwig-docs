The top-level `model_type` parameter specifies the type of model to use.

The following model types are supported:

- `ecd` (default): Encoder-Combiner-Decoder neural network model.
- `gbm`: Gradient Boosting Machine tree-based model.

```yaml
model_type: ecd
```

Every model type has trainers associated with it. See the [Trainer](../trainer) section for details about the supported training algorithms per model type.

## Model Type: ECD

See the [ECD documentation](../../user_guide/how_ludwig_works/#ecd-architecture) for details about the Encoder-Combiner-Decoder deep learning architecture.

!!! check

    The full breadth of Ludwig functionality is available for the `ecd` model type.

## Model Type: GBM

The GBM model type is a gradient boosting machine (GBM) tree model. It is a tree model that is trained using a supported tree learner. Currently, the only supported tree learner is LightGBM.

!!! attention

    Selecting the `gbm` model type introduces the following limitations:

    - dataset larger than 1GB are not supported
    - only binary, category, and number features are supported
    - only a single output feature is supported
    - the `combiner` section is ignored
