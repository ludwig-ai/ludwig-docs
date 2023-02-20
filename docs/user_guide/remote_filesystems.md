Remote filesystems / object storage systems like Amazon S3 are useful for working with large datasets or
running on a cluster of machines for [distributed training](./distributed_training.md). Ludwig
provides out-of-the-box support for reading and writing to remote systems through [fsspec](https://filesystem-spec.readthedocs.io/en/latest/).

Example:

```bash
ludwig train \
    --dataset s3://my_datasets/subdir/dataset.parquet \
    --output_directory s3://my_experiments/foo
```

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
