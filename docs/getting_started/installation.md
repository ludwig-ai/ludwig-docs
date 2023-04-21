Ludwig is a *declarative* deep learning framework that allows users to train, evaluate, and deploy models without
the need to write code.

Being declarative means you only need to tell Ludwig *what* columns in your data are input
and output features, and Ludwig will figure out *how* to train the best model.

For users familiar with Python, we recommend installing with [`pip`][pip] within an isolated
[virtual environment](https://docs.python-guide.org/dev/virtualenvs/). If not, you can use our
pre-built [`docker`][docker] images. Advanced users can also install Ludwig from [`git`][git].

For large or long-running workloads, Ludwig can be run remotely in the cloud or on a private compute cluster using [`Ray`][ray].

  [pip]: #with-pip
  [venv]: https://docs.python-guide.org/dev/virtualenvs/
  [docker]: #with-docker
  [git]: #with-git
  [ray]: /getting_started/ray.md

# Python (with Pip) <small>recommended</small> { #with-pip }

``` sh
pip install ludwig
```

This will install Ludwig's basic requirements for modeling with binary, category, number, text, image, and audio features.
The requirements for additional functionality are separated out so that users are able to install only the ones they actually need:

- `ludwig[serve]` for serving dependencies.
- `ludwig[viz]` for visualization dependencies.
- `ludwig[hyperopt]` for hyperparameter optimization dependencies.
- `ludwig[distributed]` for distributed training on [Ray](https://www.ray.io/) using [Dask](https://dask.org/) and [Horovod](https://github.com/horovod/horovod).
- `ludwig[tree]` for training [LightGBM](https://lightgbm.readthedocs.io/) models using `model_type: gbm` in the config.

 The full set of dependencies can be installed with:

 ``` sh
 pip install 'ludwig[full]'
 ```

## GPU support

If your machine has a GPU to accelerate the training process, make sure you install a GPU-enabled version of PyTorch before installing Ludwig:

``` sh
pip install torch -f https://download.pytorch.org/whl/cu118/torch_stable.html
```

The example above will install the latest version of PyTorch with CUDA 11.8. See the official [PyTorch docs](https://pytorch.org/get-started/locally/) for
more details on installing the right version of PyTorch for your environment.

# Docker { #with-docker }

The Ludwig team publishes official [Docker images](https://hub.docker.com/u/ludwigai) that come with the full set of
dependencies pre-installed. You can pull the `latest` images (for the most recent official Ludwig release) by running:

``` sh
docker pull ludwigai/ludwig:latest
```

The `ludwig` command line tool is provided as the entrypoint for all commands.

## GPU support

If your machine has a GPU to accelerate the training process, pull the official Ludwig image with GPU support:

``` sh
docker pull ludwigai/ludwig-gpu:latest
```

# Git { #with-git }

For developers who wish to build the source code from the [GitHub](https://github.com/ludwig-ai/ludwig/) repository, first clone the repo locally:

``` sh
git clone git@github.com:ludwig-ai/ludwig.git
```

Install the required dependencies:

``` sh
pip install -e '.[test]'
```

The `test` extra will pull in all Ludwig dependencies in addition to test dependencies.
