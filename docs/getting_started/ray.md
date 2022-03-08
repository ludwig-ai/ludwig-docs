[Ray](https://www.ray.io/) is a framework for distributed computing that makes it easy to scale up code that runs on your local machine to execute in parallel across a cluster.

Let's spin up a ray cluster, so we can try out distributed training and hyperparameter tuning in parallel. For this example, make sure you have access to an AWS EC2 node provider.

First install the Ray Cluster Launcher:

```commandline
pip install ray
```

Next let's make a configuration file named cluster.yaml for the Ray Cluster:

```cluster.yaml
cluster_name: ludwig-ray-gpu-nightly

min_workers: 4
max_workers: 4

docker:
    image: "ludwigai/ludwig-ray-gpu:nightly"
    container_name: "ray_container"

head_node:
    InstanceType: c5.2xlarge
    ImageId: latest_dlami

worker_nodes:
    InstanceType: g4dn.xlarge
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

For more information on using Ray with Ludwig, follow the full configuration guide [here](https://ludwig-ai.github.io/ludwig-docs/0.4/user_guide/distributed_training/#:~:text=Horovod-,Ray,-Running%20Ludwig%20with).