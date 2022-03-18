# Where can I find Ludwig's development roadmap?

Larger projects are tracked in [GitHub Projects](https://github.com/ludwig-ai/ludwig/projects).

Smaller feature requests are tracked in [Github Issues](https://github.com/ludwig-ai/ludwig/issues).

We try our best to keep these up to date, but if there's a specific feature or
model you are interested in, feel free to ping the
[Ludwig Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ).

# How can I help?

Join the [Ludwig Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ)!
Sometimes we'll organize community fixit, documentation, and bug bash efforts,
or consider taking on an [easy bug](https://github.com/ludwig-ai/ludwig/labels/easy)
to start.

# Can I use Ludwig for my project?

Yes! The people behind Ludwig are veterans of research and open source. If your
work does get published, please consider citing Ludwig and submitting
improvements back to Ludwig.

How to cite:

```
@misc{Molino2019,
  author = {Piero Molino and Yaroslav Dudin and Sai Sumanth Miryala},
  title = {Ludwig: a type-based declarative deep learning toolbox},
  year = {2019},
  eprint = {arXiv:1909.07930},
}
```

# Can I use Ludwig models in production?

Yes! Ludwig models can be exported to Neuropod, MLFlow, and Torchscript.
`ludwig serve` provides basic POST/GET serving endpoints, powered by FastAPI.

If you are interested in a more sophisticated, hosted cloud infrastructure
solution with reliable SLAs, check out Predibase.

# Does Ludwig's Architecture work for model X?

Most likely.

Ludwig's encoder-combiner-decoder framework is designed to generally mapping
some input to some output.

- Encoders parse raw input data into tensors (potentially using a model).
- Combiners combine the outputs of Input Encoders (potentially using a model).
- Decoders decode the outputs of Encoders and Combiners into output tensors
  (potentially using a model).

Decoder-only, encoder-only, encoder-decoder, vanilla feed-forward, transformers,
and more have all been implemented in Ludwig.

# What does Ludwig not support (yet)?

Domains of deep learning that Ludwig does not support (yet):

- Self-supervised learning.
- Reinforcement learning.
- Generative image and audio models (generative text models are supported).

We are actively working on supporting self-supervised learning.

# Do all datasets need to be loaded in memory?

Locally, it depends on the type of feature: image features can be dynamically
loaded from disk from an opened hdf5 file, while other types of features are
loaded entirely in memory for speed.

Ludwig supports training with very large datasets on Ray using
[Ray Datasets](https://docs.ray.io/en/latest/data/dataset.html). Read more about
using [Ludwig on Ray](../user_guide/distributed_training/#ray).

If you are interested in a premium hosted Ludwig infrastructure and APIs, with a
richer set of APIs to support modeling with large datasets, check out Predibase.

# Who develops Ludwig?

Ludwig was created in 2019 by Piero Molino, with help from Yaroslav Dudin, and
Sai Sumanth Miryala while at Uber AI.

Today, Ludwig is open source, supported by the Linux Foundation, with source
code hosted on Github.

Ludwig is actively developed and maintained by [Ludwig Maintainers](https://github.com/orgs/ludwig-ai/teams/ludwig-maintainers),
which consists mostly of staff at Predibase, and community contributors, all of
whom are listed in each of Ludwig's release notes.

Happy contributing!
