# Overview

Ludwig data preprocessing performs a few different operations on the incoming dataset:

1. **Computing metadata** like vocabulary, vocabulary size, and sequence lengths. This allows Ludwig to create
   dictionaries like `idx2str` or `str2idx` to map between raw data values to tensor values.
1. **Handling missing values** any rows/examples that have missing feature values are filled in with constants or other
   example-derived values (see [Type-Global Preprocesing Configuration](../../configuration/defaults.md#type-global-preprocessing)).
1. (optional) **Splitting dataset** into train, validation, and test based on splitting percentages, or using explicitly
   specified splits.
1. (optional) **Balancing data** which can be useful for datasets with heavily underrepresented or overrepresented
   classes.

Data preprocessing maps raw data to two files: 1) a processed dataset file containing tensors (Parquet format)
and 2) a JSON file of metadata.
The processed dataset and metadata files are saved in the [cache directory](../../configuration/backend.md) (defaults to the
same directory as the input dataset), unless `--skip_save_processed_input` is
used. The two files will serve as a cache to help avoid performing the same preprocessing again for subsequent experiments, which can be time
consuming.

The preprocessing process is highly customizable via the [Type-Global Preprocessing Section](../../configuration/defaults.md#type-global-preprocessing) of
the Ludwig config. The basic assumption is always that all data is UTF-8 encoded and contains one row for each
example and one column for each feature.

It's helpful to assign types to each feature. Some types assume a specific format, and different types will have
different ways of mapping raw data into tensors. From v0.5, users also have the option to rely on Ludwig AutoML to
assign types automatically.

# Preprocessing for different data types

Each datatype is preprocessed in a different way, using different parameters and different [tokenizers](../../configuration/defaults.md#tokenizers).
Details on how to set those parameters for each feature type and for each specific feature is described in the
[Configuration - Defaults - Type-Global Preprocessing](../../configuration/defaults.md#type-global-preprocessing) section.

## Binary features

`Binary` features are directly transformed into a binary valued vector of length `n` (where `n` is the size of the
dataset) and added to the processed dataset with a key that reflects the name of column in the dataset. No additional information
about them is available in the JSON metadata file.

## Number features

`Number` features are directly transformed into a float valued vector of length `n` (where `n` is the size of the
dataset) and added to the processed dataset with a key that reflects the name of column in the dataset. No additional information
about them is available in the JSON metadata file.

## Category features

`Category` features are transformed into an integer valued vector of size `n` (where `n` is the size of the dataset)
and added to the processed dataset with a key that reflects the name of column in the dataset.

The way categories are mapped into integers consists of first collecting a dictionary of all the unique category
strings present in the column of the dataset, then rank them by frequency and then assign them an increasing integer ID
from the most frequent to the most rare (with 0 being assigned to a `<UNK>` token).  The column name is added to the
JSON file, with an associated dictionary containing:

1. the mapping from integer to string (`idx2str`)
1. the mapping from string to id (`str2idx`)
1. the mapping from string to frequency (`str2freq`)
1. the size of the set of all tokens (`vocab_size`)
1. additional preprocessing information (by default how to fill missing values
   and what token to use to fill missing values)

## Set features

`Set` features are transformed into a binary (int8 actually) valued matrix of size `n x l` (where `n` is the size of the
dataset and `l` is the minimum of the size of the biggest set and a `max_size` parameter) and added to processed dataset with a key
that reflects the name of column in the dataset.

The way sets are mapped into integers consists in first using a tokenizer to map from strings to sequences of set items
(by default this is done by splitting on spaces).  Then a dictionary of all the different set item strings present in
the column of the dataset is collected, then they are ranked by frequency and an increasing integer ID is assigned to
them from the most frequent to the most rare (with 0 being assigned to `<PAD>` used for padding and 1 assigned to
`<UNK>` item).  The column name is added to the JSON file, with an associated dictionary containing:

1. the mapping from integer to string (`idx2str`)
1. the mapping from string to id (`str2idx`)
1. the mapping from string to frequency (`str2freq`)
1. the maximum size of all sets (`max_set_size`)
1. additional preprocessing information (by default how to fill missing values
   and what token to use to fill missing values)

## Bag features

`Bag` features are treated in the same way of set features, with the only difference being that the matrix had float
values (frequencies).

## Sequence Features

Sequence features by default are managed by `space` tokenizers. This splits the content of the feature value into a list
of strings using space.

| before tokenizer       | after tokenizer            |
| ---------------------- | -------------------------- |
| "token3 token4 token2" | \[token3, token4, token2\] |
| "token3 token1"        | \[token3, token1\]         |

Computing metadata: A list `idx2str` and two dictionaries `str2idx` and `str2freq` are created containing all the tokens
in all the lists obtained by splitting all the rows of the column and an integer id is assigned to each of them (in
order of frequency).

```json
{
    "column_name": {
        "idx2str": [
            "<PAD>",
            "<UNK>",
            "token3",
            "token2",
            "token4",
            "token1"
        ],
        "str2idx": {
            "<PAD>": 0,
            "<UNK>": 1,
            "token3": 2,
            "token2": 3,
            "token4": 4,
            "token1": 5
        },
        "str2freq": {
            "<PAD>":  0,
            "<UNK>":  0,
            "token3": 2,
            "token2": 1,
            "token4": 1,
            "token1": 1
        }
    }
}
```

Finally, a numpy matrix is created with sizes `n x l` where `n` is the number of rows in the column and `l` is the
minimum of the longest tokenized list and a `max_length` parameter that can be set. All sequences shorter than `l` are
right-padded to the `max_length` (though this behavior may also be modified through a parameter).

| after tokenizer            | numpy matrix |
| -------------------------- | ------------ |
| \[token3, token4, token2\] | 2 4 3        |
| \[token3, token1\]         | 2 5 0        |

The final result matrix is saved in the processed dataset with the name of the original column in the dataset as key, while the
mapping from token to integer ID (and its inverse mapping) is saved in the JSON file.

A frequency-ordered vocabulary dictionary is created which maps tokens to integer IDs. Special symbols like `<PAD>`,
`<START>`, `<STOP>`, and `<UNK>` have specific indices. By default, we use `[0, 1, 2, 3]`, but these can be overridden
manually.

If a `huggingface` encoder is specified, then that encoder's special symbol indices will be used instead.

The computed metadata includes:

1. the mapping from integer to string (`idx2str`)
1. the mapping from string to id (`str2idx`)
1. the mapping from string to frequency (`str2freq`)
1. the maximum length of all sequences (`max_sequence_length`)
1. additional preprocessing information (by default how to fill missing values
   and what token to use to fill missing values)

## Text features

`Text` features are treated in the same way of sequence features, with a couple differences. Two different tokenizations
happen, one that splits at every character and one that uses a custom tokenizer. Two different keys are added to the
processed dataset file, one for the matrix of characters and one for the matrix of symbols.

The same thing happens in the JSON file, where there are two sets of dictionaries, one for mapping characters to
integers (and the inverse) and symbols to integers (and their inverse).

If a `huggingface` encoder is specified, then that encoder's tokenizer will be used for the symbol-based tokenizer.

In the configuration users can specify which level of representation to use: the character level or the symbol level.

## Timeseries features

`Timeseries` features are treated in the same way of sequence features, with the only difference being that the matrix
in the processed dataset file does not have integer values, but float values. The JSON file has no additional mapping information.

## Image features

`Image` features are transformed into a float32 valued tensor of size `n x c x h x w` (where `n` is the size of the
dataset, `c` is the number of color channels, and `h x w` is the height and width) and added to the processed dataset
with a key that reflects the name of column in the dataset.

The column name is added to the JSON file, with an associated dictionary containing preprocessing information about the
sizes of the resizing.

By default, Ludwig uses **lazy preprocessing** for image features: instead of decoding and storing all images upfront,
only file paths are stored in the processed dataset. Images are decoded on-the-fly per batch during training.
See [Lazy Preprocessing](#lazy-preprocessing-for-audio-and-image) below for details.

## Audio features

`Audio` features are transformed into float32 tensors whose shape depends on the configured feature type (raw waveform,
STFT magnitude, FBANK filterbank, etc.) and added to the processed dataset.

By default, Ludwig uses **lazy preprocessing** for audio features: file paths are stored instead of decoded tensors,
and audio is decoded per batch during training.
See [Lazy Preprocessing](#lazy-preprocessing-for-audio-and-image) below for details.

# Lazy Preprocessing for Audio and Image

Ludwig offers three preprocessing modes for audio and image features, controlled by the `mode`
parameter in the feature's `preprocessing` config.

## Choosing a Preprocessing Mode

| Mode | Memory | Preprocessing speed | Training epoch 1 | Training epoch 2+ | Best for |
|------|--------|---------------------|------------------|---------------------|----------|
| `eager` | High (O(N×tensor)) | Slow | Fast | Fast | Small datasets that fit in RAM |
| `lazy` | Low (O(batch)) | Fast | Slow (decode-bound) | Slow | Large datasets, GPU work > decode time |
| `lazy_cached` | Low (O(batch)) | Fast | Fast (GPU pipelined) | Very fast (memmap) | Large datasets, any GPU speed |

`lazy` is the default for both audio and image features.

## `lazy_cached` Mode — How It Works

`lazy_cached` combines the low preprocessing memory of `lazy` with the training speed of `eager`
after the first epoch.

```
Epoch 1 (decode + cache write)
───────────────────────────────────────────────────────────
  batch indices
       │
       ▼
  ThreadPoolExecutor                 numpy memmap
  decode(path_0..N)  ─── write ───► col_decoded_nN_..._f32.npy
       │
       ▼ (decoded tensors)
  GPU forward pass

Epoch 2+ (memmap read)
───────────────────────────────────────────────────────────
  batch indices
       │
       ▼
  memmap[indices]   (~0.1 ms/batch, no thread pool)
       │
       ▼ (decoded tensors)
  GPU forward pass
```

Once every sample has been written, a `.done` sentinel file is created next to the memmap.
`RandomAccessBatcher.set_epoch` detects this via `dataset.is_fully_cached()` and automatically
disables prefetch for all subsequent epochs.

If the `.done` file already exists (e.g. after a resumed run), the memmap is opened immediately in
read-only mode — epoch 1 behaves as fast as epoch 2+.

## `prefetch_size` Configuration

`prefetch_size` controls the depth of the background producer thread queue that pipelines batch
decode with GPU work.

| Value | Behaviour |
|-------|-----------|
| `null` (default) | Auto: 0 for `eager`, 4 for `lazy` / `lazy_cached` epoch 1, 0 for `lazy_cached` epoch 2+ |
| `0` | Disable prefetch (synchronous decode; useful for debugging) |
| `N > 0` | Use exactly N slots in the prefetch queue |

Override when needed:

- **Set to `0`** to debug decode errors without background thread noise.
- **Increase above 4** when storage is very slow (e.g. spinning disks, network filesystems) and
  the GPU consistently idles waiting for decode.

## How It Works (lazy mode)

During preprocessing, Ludwig stores file paths (strings) in the processed dataset instead of decoded
tensors.  When training begins, `PandasDataset` wraps each path column in a `LazyColumn` — an
array-compatible object that decodes a batch of files in a `ThreadPoolExecutor` whenever
`__getitem__` is called.  The thread pool runs concurrently with the GPU forward pass, so I/O
and compute overlap automatically.

```
Preprocessing                Training (per step)
─────────────────            ────────────────────────────────────────────
CSV / HF dataset             LazyColumn.__getitem__(batch_indices)
     │                            │
     ▼                            ├─ thread 1: torchaudio.load(path_0) ──┐
store paths in Parquet       │   ├─ thread 2: torchaudio.load(path_1) ──┤ → stacked
(< 1 KB per sample)          │   └─ thread N: torchaudio.load(path_N) ──┘   tensor
```

## Memory Savings in Practice

The table below compares peak heap usage for preprocessing 1,000 synthetic samples measured with
tracemalloc on a single machine:

| Modality | Eager peak | Lazy peak | Reduction |
|---|---|---|---|
| Images (64×64 RGB PNG) | ~600 MB | ~2 MB | **281×** |
| Audio (2 s / 16 kHz WAV) | ~15 MB | ~14 MB | ~1× |

Audio savings during preprocessing are modest because fbank/STFT features are compact — the main
benefit for audio is that the *training* memory footprint stays bounded, which matters when clips are
long (> 10 s) or the dataset is very large (> 100 k samples).

## HuggingFace Streaming Datasets

HuggingFace audio and image columns deliver in-memory objects rather than paths:

- **Audio**: `{"array": np.ndarray, "sampling_rate": int, "path": str | None}`
- **Image**: `PIL.Image.Image` (with or without `.filename`)

Ludwig handles both cases transparently. If the dict or PIL image already points to an existing
local file (HF's disk cache), that path is reused directly. Otherwise, Ludwig writes a WAV or PNG
file to the lazy cache directory before training:

```
~/.cache/ludwig/lazy_media/<feature_name>/
```

The cache is persistent and idempotent — re-running with the same dataset skips the write step.

### Supported In-Memory Input Types

| Type | How Ludwig handles it |
|---|---|
| `dict` with existing `"path"` | Reuses the existing file |
| `dict` with `"array"` / `"sampling_rate"` | Writes WAV to cache |
| `torch.Tensor` (audio) | Writes WAV to cache using feature's sample rate |
| `PIL.Image` with `.filename` | Reuses the existing file |
| `PIL.Image` without `.filename` | Writes PNG to cache |
| `bytes` (encoded image) | Decodes and writes PNG to cache |
| `np.ndarray` HWC or CHW | Writes PNG to cache |
| HF Image dict with `"bytes"` | Decodes and writes PNG to cache |
| HF Image dict with `"path"` | Reuses the existing file |

## Controlling the Cache Directory

`lazy_cache_dir` controls the *file* cache for in-memory sources (HuggingFace datasets that deliver
`dict` / `PIL.Image` / `bytes` instead of local paths). Ludwig writes WAV or PNG files here so that
subsequent training runs can reuse them without re-writing.

The decoded **memmap** for `lazy_cached` mode is placed next to the Parquet cache (i.e. the same
directory as `data_train_*.parquet`), not inside `lazy_cache_dir`.  If no Parquet cache exists,
the memmap falls back to `~/.cache/ludwig/lazy_media/`.

To override the file cache directory per feature:

```yaml
input_features:
  - name: speech
    type: audio
    preprocessing:
      mode: lazy_cached
      lazy_cache_dir: /fast/nvme/my_project/audio_cache

  - name: photo
    type: image
    preprocessing:
      mode: lazy_cached
      lazy_cache_dir: /fast/nvme/my_project/image_cache
```

Each feature gets its own subdirectory (`<lazy_cache_dir>/<feature_name>/`) so that multiple
features sharing the same root never collide.

## Mode Configuration Examples

All three modes shown side-by-side:

```yaml
input_features:
  # Eager — decode everything upfront; fast training, high preprocessing memory
  - name: small_audio
    type: audio
    preprocessing:
      mode: eager

  # Lazy (default) — decode per batch; low memory, GPU may idle on slow storage
  - name: large_audio
    type: audio
    preprocessing:
      mode: lazy
      prefetch_size: null      # auto (4)

  # Lazy-cached — decode + cache on epoch 1; memmap read from epoch 2+
  - name: huge_audio
    type: audio
    preprocessing:
      mode: lazy_cached
      prefetch_size: null      # auto (4 for epoch 1, 0 for epoch 2+)
      lazy_cache_dir: /fast/nvme/cache
```

!!! note
    Lazy preprocessing is automatically disabled for image features that use a **TorchVision
    pretrained encoder** (e.g. `resnet`, `efficientnet`, `vit`). Those encoders apply their own
    normalization pipeline which requires images to be decoded upfront.

## Lazy Preprocessing with the Ray Backend

When training with the Ray backend (`trainer.type: ray`), Ludwig attaches `map_batches` decode
transforms directly to the Ray dataset rather than wrapping columns in `LazyColumn` objects.
This mirrors the local behaviour while keeping decoding distributed across workers.

!!! warning
    **`lazy_cached` mode is not supported with the Ray backend.** `CachedLazyColumn` writes to a
    local numpy memmap, which is not accessible across Ray workers on different nodes.  Use `mode:
    lazy` (the default) when training with Ray.  The Ray backend has its own efficient decode
    pipeline via `map_batches` and does not benefit from local memmaps.

### How It Works in Distributed Training

```
Preprocessing                    Training (per worker, per batch)
─────────────────                ────────────────────────────────────────────────
CSV / HF dataset                 Ray dataset.iter_batches(batch_size=B)
     │                                │
     ▼                                ▼
store paths in Parquet           map_batches(lazy_decode_<col>)
(< 1 KB per sample)                   │
     │                                ├─ thread 1: decode(path_0) ─┐
     ▼                                ├─ thread 2: decode(path_1) ─┤ → stacked tensor
saved to object store            │   └─ thread N: decode(path_N) ─┘   to GPU
```

When `RayDataset.initialize_batcher()` is called:

1. The Ray dataset is **materialized** — path strings (small, < 1 KB each) are loaded into
   the Ray object store once per epoch instead of re-reading Parquet on every epoch.
2. `_with_lazy_decode()` attaches a `map_batches` transform for each lazy feature.
   The transform is a `ThreadPoolExecutor`-based decode function that runs inside the Ray
   worker pulling each block from the object store.
3. When `iter_batches` iterates blocks, the decode transform fires in-process, so each worker
   decodes only the audio/image files needed for its current block.
4. The prediction path also applies lazy decode transforms — path strings in the dataset are
   decoded before being passed to the model (previously a bug caused raw paths to reach the
   tensor cast step).

### Memory Implications

| | Eager (`mode: eager`) | Lazy (`mode: lazy`, default) |
|---|---|---|
| **Object store per worker** | O(N × tensor_size) | O(N × path_size) ≈ O(N × 1 KB) |
| **Worker heap per batch** | O(batch_size × tensor_size) | O(batch_size × tensor_size) |
| **Total cluster memory** | proportional to dataset size | dominated by model + one batch |

With `mode: lazy`, the object store holds only path strings (< 1 KB each). Each worker decodes
`batch_size` files at a time into its own heap — there is no spike proportional to the full
dataset size across the cluster.

With `mode: eager`, decoded tensors (potentially hundreds of MB each for long audio clips) are
stored in the Ray object store and replicated to each worker. For large datasets this can exhaust
object store memory, causing Ray to spill to disk or OOM.

### File Accessibility

Because decoding happens inside the Ray workers (which may run on different nodes in a cluster),
each worker must be able to read the audio/image files directly. This means:

- **Single-node Ray clusters**: no extra setup needed — all workers share the local filesystem.
- **Multi-node Ray clusters**: files must be accessible from every node. Use a shared filesystem
  (NFS, Lustre, GPFS) or an object store (S3, GCS, Azure Blob Storage). The path stored in the
  Parquet dataset must be resolvable from every node.

!!! tip
    Use `lazy_cache_dir` to point to a shared filesystem path when running multi-node training.
    This ensures the HuggingFace lazy cache (if used) is also visible to all workers.

### Configuration

Lazy preprocessing works transparently with the Ray backend — no extra configuration is needed:

```yaml
trainer:
  type: ray

input_features:
  - name: audio
    type: audio
    preprocessing:
      mode: lazy          # default — decode in worker, low memory

  - name: image
    type: image
    preprocessing:
      mode: lazy          # default — decode in worker, low memory
```

To force all decoding to happen before distributed training (useful when files are on a slow
network and you have enough cluster memory):

```yaml
input_features:
  - name: audio
    type: audio
    preprocessing:
      mode: eager         # decode upfront; tensors stored in Ray object store
```
