Ludwig has strong support for [Ray](https://www.ray.io/), a framework for distributed computing that makes it easy to scale up code that runs on your local machine to execute in parallel across a cluster of machines.

Let's spin up a ray cluster, so we can try out distributed training and hyperparameter tuning in parallel. Make sure you have access to an AWS EC2 node provider.

First install the Ray Cluster Launcher:

```commandline
pip install ray
```

Next let's make a configuration file named `cluster.yaml` for the Ray Cluster:

```yaml  title="cluster.yaml"
cluster_name: ludwig-ray-gpu-latest

min_workers: 4
max_workers: 4

docker:
    image: "ludwigai/ludwig-ray-gpu:latest"
    container_name: "ray_container"

head_node:
    InstanceType: m5.2xlarge
    ImageId: latest_dlami

worker_nodes:
    InstanceType: g4dn.2xlarge
    ImageId: latest_dlami
```

Finally, you can spin up the cluster with the following command:

```sh
ray up cluster.yaml
```

In order to run a distributed training job, make sure you have your dataset stored in an S3 bucket, and run this command:

```sh
ray submit cluster.yaml ludwig train --config rotten_tomatoes.yaml --dataset s3://mybucket/rotten_tomatoes.csv
```

You can also run a distributed hyperopt job with this command:

```sh
ray submit cluster.yaml ludwig hyperopt --config rotten_tomatoes.yaml --dataset s3://mybucket/rotten_tomatoes.csv
```

For more information on using Ray with Ludwig, refer to the [ray configuration guide](../../user_guide/distributed_training).
