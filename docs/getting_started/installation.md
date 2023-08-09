# Use pip

For users familiar with Python, we recommend installing with [`pip`][pip] within an isolated
[virtual environment](https://docs.python-guide.org/dev/virtualenvs/). If not, you can use our
pre-built [`docker`](https://hub.docker.com/search?q=ludwigai) images.

``` sh
pip install ludwig
```

For large or long-running workloads, Ludwig can be run remotely in the cloud or on a private compute cluster using [`Ray`](/getting_started/ray).

## Install additional packages

Optional Ludwig functionality is separated out into subpackages. Install what you need:

- `ludwig[llm]` for LLM dependencies.
- `ludwig[serve]` for serving dependencies.
- `ludwig[viz]` for visualization dependencies.
- `ludwig[hyperopt]` for hyperparameter optimization dependencies.
- `ludwig[distributed]` for distributed training on [Ray](https://www.ray.io/) using [Dask](https://dask.org/).
- `ludwig[explain]` for prediction explanations.
- `ludwig[tree]` for LightGBM and tree-based models.
- `ludwig[test]` for running ludwig's integration and unit tests.
- `ludwig[benchmarking]` for Ludwig model benchmarking.
- `ludwig[full]` for the full set of dependencies.

## Install from git

```sh
pip install git+https://github.com/ludwig-ai/ludwig.git
```

## Install from source

```sh
git clone https://github.com/ludwig-ai/ludwig.git
cd ludwig
pip install -e .
```

# Use devcontainers

Ludwig supports development on [VSCode devcontainers](https://code.visualstudio.com/docs/devcontainers/containers). See Ludwig's [devcontainer files](https://github.com/ludwig-ai/ludwig/tree/master/.devcontainer).

# Use pre-build docker images

See Ludiwg's [docker docs](/getting_started/docker).
