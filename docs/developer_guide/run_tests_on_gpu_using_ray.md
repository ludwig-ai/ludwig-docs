As part of Ludwig's GitHub actions PR checks, all Ludwig tests must pass with
and without GPU availability.

To debug a specific test on GPU, it may be useful to run Ludwig GPU tests using
Ray.

# Setup

## 1. Set up an AWS AMI with a GPU

Reach out to your AWS account administrator, or [set up an account for yourself](https://aws.amazon.com/getting-started/hands-on/get-started-dlami/).

## 2. Test if you have the AWS CLI

```sh
aws s3 ls
```

If not, install it from [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

## 3. Set up AWS keys

1. AWS Credentials [you will need to set this up for Ray to authenticate you]

    [How to create AWS Access Key ID](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-creds)

    Once created, download your access key so you can refer to it.

2. Run `aws configure` to configure your AWS CLI with your access credentials

    [Configuration and credential file settings - AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)

3. (optional) Get an AWS PEM file

    Not needed for unit tests on GPU, which never spins up new nodes, but it
    will be needed if you ever want to enable Ray to launch new nodes.

    [Amazon EC2 key pairs and Linux instances - Amazon Elastic Compute Cloud](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair)

## 4. Get Ray

Install ray locally:

```sh
pip install -U "ray[default]" boto3
```

## 5. Set up a Ray Config

```sh
vim $HOME/.clusters/cluster.yaml
```

Copy the sample ray config below and edit all the `<...>` values to match your
local dev environment.

```yaml
cluster_name: <$USER>-ludwig-ray-g4dn

max_workers: 3

docker:
  image: "ludwigai/ludwig-ray-gpu:master"
  container_name: "ray_container"
  pull_before_run: True
  run_options: # Extra options to pass into "docker run"
    - --ulimit nofile=65536:65536

provider:
  type: aws
  region: <us-east-2>
  availability_zone: <us-east-2a>

available_node_types:
  ray.head.default:
    resources: {}
    node_config:
      InstanceType: g4dn.4xlarge
      ImageId: latest_dlami
      BlockDeviceMappings:
        - DeviceName: /dev/sda1
          Ebs:
            VolumeSize: 100
  ray.worker.default:
    min_workers: 0
    max_workers: 0
    resources: {}
    node_config:
      InstanceType: g4dn.4xlarge
      ImageId: latest_dlami

head_node_type: ray.head.default

file_mounts:
  {
    /home/ubuntu/ludwig/: </Users/$USER/ludwig>,  # Ludwig Repo.
    /home/ray/.aws: </Users/$USER/.aws>,  # AWS credentials.
  }

rsync_exclude:
  - "**/.git"
  - "**/.git/**"

rsync_filter:
  - ".gitignore"

setup_commands:
  - pip uninstall -y ludwig && pip install -e /home/ubuntu/ludwig/.
  - pip install s3fs==2021.10.0 aiobotocore==1.4.2 boto3==1.17.106
  - pip install pandas==1.1.4
  - pip install hydra-core --upgrade

head_start_ray_commands:
  - ray stop --force
  - ray start --head --port=6379 --object-manager-port=8076 --autoscaling-config=~/ray_bootstrap_config.yaml

worker_start_ray_commands:
  - ray stop --force
  - ray start --address=$RAY_HEAD_IP:6379 --object-manager-port=8076
```

Set an environment variable mapping to the location (can be relative) of your
cluster config:

```sh
export CLUSTER="$HOME/.clusters/cluster.yaml"
```

# Developer Workflow

## (once) Launch the ray cluster

export CLUSTER="$HOME/cluster_g4dn.yaml"
export CLUSTER_CPU="$HOME/cluster_cpu.yaml"
ray up $CLUSTER

## Make local changes

Run tests locally.

```sh
pytest tests/...
```

## Rsync your local changes to the ray GPU cluster

```
ray rsync_up $CLUSTER -A '/Users/$USER/ludwig/' '/home/ubuntu/ludwig'
ray rsync_up $CLUSTER_CPU -A '/Users/$USER/ludwig/' '/home/ubuntu/ludwig'
```

!!! warning

    The trailing backslash `/` is important!

## Run tests on the GPU cluster from the Ray-mounted ludwig directory

```sh
ray exec $CLUSTER "cd /home/ubuntu/ludwig && pytest tests/"
```

You can also connect directly to a terminal on the cluster head:

```sh
ray attach $CLUSTER
```
