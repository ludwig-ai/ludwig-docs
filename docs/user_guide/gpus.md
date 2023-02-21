Ludwig will automatically detect and run on GPU hardware when available. To run on multiple GPUs or a cluster of GPU machines,
see [Distributed Training](./distributed_training/index.md).

GPUs will dramatically improve the speed of training using the default ECD architecture. Some features types, nmely text and images, de facto require GPUs to train in a reasonable amount of time. That said, if you are only using tabular features (binary, category, number) with the default concat combiner, or are training a GBM model, you may not notice a difference without GPUs, so the utility will vary by use case.

## Running on Apple Metal

Ludwig supports experimental support for Apple Metal GPUs with the [Metal Performance Shaders (MPS)](https://developer.apple.com/metal/pytorch/) library. To try it out, set the `LUDWIG_ENABLE_MPS` environment variable when training:

```bash
LUDWIG_ENABLE_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 ludwig train ...
```

It is also recommended to set `PYTORCH_ENABLE_MPS_FALLBACK=1` as well, as not all operations used by Ludwig are supported by MPS.

Empircally, we've observed significant speedups using MPS when training on larger text and image models. However, we also observed
performance decreases using smaller tabular models, which is why we do not enable this feature by default. We recommend trying with and
without to see which gives the best performance for your use case.

## Tips

### Avoiding CUDA OOMs

GPUs can easily run out of memory when training on large models. To avoid these
types of errors, we recommend trying the following in order:

- Setting `trainer.batch_size=auto` in the config (this is the current default).
- Manually setting `trainer.max_batch_size`.
- Trying a smaller model architecture.
- Splitting the model across multiple GPUs the [Fully Sharded Data Parallel](../configuration/backend.md#fully-sharded-data-parallel-fsdp) strategy.

### Disabling GPUs

You can disable GPU acceleration during training by setting `CUDA_VISIBLE_DEVICES=""` in the environment. This can be useful for debugging runtime failures specific to your GPU hardware.