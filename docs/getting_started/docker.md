You can also run Ludwig using the docker images available on [dockerhub](https://hub.docker.com/search?q=ludwigai). 
These images come with a full set of pre-requiste packages to support the capabilities of Ludwig

### Repositories

The following repositories each contain a version of Ludwig with full features built
from the `master` branch.

- `ludwigai/ludwig` Ludwig packaged with PyTorch
- `ludwigai/ludwig-gpu` Ludwig packaged with gpu-enabled version of PyTorch
- `ludwigai/ludwig-ray` Ludwig packaged with PyTorch
  and [nightly build of ray-project/ray](https://github.com/ray-project/ray)
- `ludwigai/ludwig-ray-gpu` Ludwig packaged with gpu-enabled versions of PyTorch
  and [nightly build of ray-project/ray](https://github.com/ray-project/ray)

### Image Tags

The following are the image tags that can be used when pulling and running the docker images.

- `master` - built from Ludwig's `master` branch
- `nightly` - nightly build of Ludwig's software.
- `sha-<commit point>` - version of Ludwig software at designated git sha1
  7-character commit point.

## Running Containers

Here are some examples of using the `ludwigai/ludwig:master` image to:

- run the `ludwig cli` command or
- run Python program containing Ludwig api or
- view Ludwig results with Tensorboard

For purposes of the examples assume this host directory structure

```
/top/level/directory/path/
    data/
        train.csv
    src/
        config.yaml
        ludwig_api_program.py
```

### Run Ludwig CLI

```shell
# set shell variable to parent directory
parent_path=/top/level/directory/path

# invoke docker run command to execute the ludwig cli
# map host directory ${parent_path}/data to container /data directory
# map host directory ${parent_path}/src to container /src directory
docker run -v ${parent_path}/data:/data  \
    -v ${parent_path}/src:/src \
    ludwigai/ludwig:master \
    experiment --config /src/config.yaml \
        --dataset /data/train.csv \
        --output_directory /src/results
```

Experiment results can be found in host
directory `/top/level/directory/path/src/results`

### Run Python program using Ludwig APIs

```shell
# set shell variable to parent directory
parent_path=/top/level/directory/path

# invoke docker run command to execute Python interpreter
# map host directory ${parent_path}/data to container /data directory
# map host directory ${parent_path}/src to container /src directory
# set current working directory to container /src directory
# change default entrypoint from ludwig to python
docker run  -v ${parent_path}/data:/data  \
    -v ${parent_path}/src:/src \
    -w /src \
    --entrypoint python \
    ludwigai/ludwig:master /src/ludwig_api_program.py
```

Ludwig results can be found in host
directory `/top/level/directory/path/src/results`