Cloud object storage systems like Amazon S3 are useful for working with large datasets or
running on a cluster of machines for [distributed training](./distributed_training/index.md). Ludwig
provides out-of-the-box support for reading and writing to remote systems through [fsspec](https://filesystem-spec.readthedocs.io/en/latest/).

Example:

```bash
ludwig train \
    --dataset s3://my_datasets/subdir/dataset.parquet \
    --output_directory s3://my_experiments/foo
```

# Environment Setup

The sections below cover how to read and write between your preferred remote filesystem in Ludwig.

## Amazon S3

Install filesystem driver in your Docker image: `pip install s3fs`.

Mount your `$HOME/.aws/credentials` file into the container or set the following enviroment variables:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Refer to paths with protocol `s3://`.

### MinIO

MinIO uses the same protocol as s3, but requires an additional environment variable to be set:

- `AWS_ENDPOINT_URL`

## Google Cloud Storage

Install filesystem driver in your Docker image: `pip install gcsfs`.

Generate a token as described [here](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#iam-service-account-keys-create-console).

Mount the token file into the container at one of the locations described in the `gcsfs` [docs](https://gcsfs.readthedocs.io/en/latest/#credentials).

Refer to paths with protocol `gs://` or `gcs://`.

## Azure Storage

Install filesystem driver in your Docker image: `pip install adlfs`.

Mount your `$HOME/.aws/credentials` file into the container or set the following enviroment variables:

- `AZURE_STORAGE_CONNECTION_STRING`

See `adlfs` [docs](https://github.com/fsspec/adlfs#setting-credentials) for more details.

Refer to paths with protocol `az://` or `abfs://`.

# Additional Configuration

## Remote Dataset Cache

Often your input datasets will be in a read-only location such as a shared data lake. In these cases, you won't want to
rely on Ludwig's default caching behavior of writing to the same base directory as the input dataset. Instead, you can configure
Ludwig to write to a dedicated cache directory / bucket by configuring the `backend` section of the config:

```yaml
backend:
  cache_dir: "s3://ludwig_cache"
```

Individual entries will be written using a filename computed from the checksum of the dataset and Ludwig config used for training.

One additional benefit of setting up a dedicated cache is to make use of cache eviction policies. For example, setting up a TTL 
so cached datasets are automatically cleaned up after a few days.

### Using different cache and dataset filesystems

In some cases you may want your dartaset cache to reside in a different filesystem or account than your input dataset.
Since this requires maintaing two sets of credentials -- one of the input data and one for the cache -- Ludwig provides
additional configuration options for the cache credentials.

Credentials can be provided explicitly in the config:

```yaml
backend:
  cache_dir: "s3://ludwig_cache"
  cache_credentials:
    s3:
      client_kwargs:
        aws_access_key_id: "test"
        aws_secret_access_key: "test"
```

Or in a mounted file for additional security:

```yaml
backend:
  cache_dir: "s3://ludwig_cache"
  cache_credentials: /home/user/.credentials.json
```
