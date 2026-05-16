{% from './macros/includes.md' import render_fields, render_yaml %}
{% set mv_details = "See [Missing Value Strategy](./input_features.md#missing-value-strategy) for details." %}
{% set type = "See explanations for each type [here](audio_features.md#input-features)." %}
{% set details = {"missing_value_strategy": mv_details, "type": type} %}

# Preprocessing

Example of a preprocessing specification (assuming the audio files have a sample rate of 16000):

{% set preprocessing = get_feature_preprocessing_schema("audio") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

Ludwig supports reading audio files using PyTorch's [Torchaudio](https://pytorch.org/audio/stable/index.html) library. This library supports `WAV`, `AMB`, `MP3`, `FLAC`, `OGG/VORBIS`, `OPUS`, `SPHERE`, and `AMR-NB` formats.

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing), details=details) }}

Preprocessing parameters can also be defined once and applied to all audio input features using the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

## Lazy Preprocessing

By default (`mode: lazy`), Ludwig does **not** decode audio files during the preprocessing phase.
Instead, it stores file paths in the processed dataset and decodes audio clips on-the-fly, one batch
at a time, during training. This approach has significant memory advantages for large audio datasets.

### Why Lazy Preprocessing?

The traditional eager approach decodes every audio file upfront and stores the resulting tensors in
the processed dataset (Parquet file). For a dataset of N clips each lasting L seconds at sampling
rate S, the peak preprocessing memory is roughly:

```
N × L × S × 4 bytes  (float32)
```

For 100,000 clips at 5 seconds / 16 kHz, that is ~32 GB — likely exceeding available RAM before
training even starts.

With lazy preprocessing, peak memory during preprocessing drops to near zero for the audio tensors
(only paths are stored). During training, only one batch of decoded audio lives in memory at a time,
so peak memory is:

```
batch_size × L × S × 4 bytes
```

Decoding happens in a `ThreadPoolExecutor` that runs in parallel with the GPU forward pass, so
throughput is not meaningfully affected compared to the eager path.

### Preprocessing Modes

Ludwig supports three preprocessing modes, controlled by the `mode` parameter:

| Mode | Behaviour |
|------|-----------|
| `lazy` (default) | Stores file paths; decodes one batch at a time during training. Lowest memory footprint. |
| `eager` | Decodes all files during preprocessing and stores the resulting tensors in the Parquet cache. Fastest training once preprocessing is done, but requires enough RAM to hold the entire decoded dataset. |
| `lazy_cached` | Behaves like `lazy` on the first training epoch (decoding each sample once and writing it to a numpy memmap alongside the Parquet cache). Subsequent epochs read from the memmap directly, eliminating per-batch decode overhead while keeping peak memory bounded during preprocessing. |

### Configuration

```yaml
input_features:
  - name: audio
    type: audio
    preprocessing:
      mode: lazy               # "lazy" (default), "eager", or "lazy_cached"
      prefetch_size: null      # null = auto (0 for eager, 4 for lazy/lazy_cached epoch 1)
      lazy_cache_dir: null     # where to cache WAV files when source is in-memory (HF datasets)
      audio_file_length_limit_in_s: 7.5
      type: fbank
      num_filter_bands: 80
```

#### `prefetch_size`

Controls how many batches are decoded in a background thread while the GPU processes the current
batch. `null` (default) selects automatically:

- `0` for `eager` mode (tensors already in memory — no prefetch needed)
- `4` for `lazy` and the first epoch of `lazy_cached`
- `0` automatically after epoch 1 in `lazy_cached` mode once the memmap is fully written (memmap
  reads are fast enough that background pipelining adds no measurable benefit)

Set to `0` to disable prefetch entirely, or to a positive integer to override.

#### `lazy_cached` — persistent decode cache

`lazy_cached` is ideal when you want to pay the decode cost once and get near-eager training speed
from epoch 2 onward. The decoded memmap is placed next to the Parquet cache file:

```
<parquet_cache_dir>/<proc_col>_decoded_n<N>_<shape>_f32.npy
```

If the cache file already exists (from a previous run), the first epoch also reads from it directly.

```yaml
input_features:
  - name: audio
    type: audio
    preprocessing:
      mode: lazy_cached
      audio_file_length_limit_in_s: 7.5
      type: fbank
      num_filter_bands: 80
```

### Lazy Preprocessing with HuggingFace Datasets

When loading a HuggingFace dataset (e.g. `datasets.load_dataset(...)`), audio columns are delivered
as Python dicts — not file paths:

```python
{
    "array": np.ndarray,          # decoded waveform, shape (samples,)
    "sampling_rate": 16000,       # sample rate in Hz
    "path": "/path/to/cache.wav"  # optional: HF's local cache path
}
```

Ludwig handles this transparently:

1. **If `path` points to an existing file on disk** (HuggingFace's local cache), Ludwig reuses that
   file directly — no copy is made.
2. **Otherwise**, Ludwig writes the waveform to a WAV file in `lazy_cache_dir` and uses that path.

This means that for most HuggingFace audio datasets, the first run caches files to
`~/.cache/ludwig/lazy_media/<feature_name>/` and subsequent runs skip the write step entirely
(the cache is persistent and idempotent).

### Controlling the File Cache Directory

`lazy_cache_dir` controls where WAV files are written when the source data is **in-memory** (e.g.
a HuggingFace dataset). It has no effect when the input column already contains local file paths.

By default, cached WAV files are written to:

```
~/.cache/ludwig/lazy_media/<feature_name>/
```

To use a different location — for example, a fast NVMe drive or a shared network path — set
`lazy_cache_dir` in the preprocessing config:

```yaml
input_features:
  - name: speech
    type: audio
    preprocessing:
      mode: lazy
      lazy_cache_dir: /fast/nvme/my_project/audio_cache
```

The per-feature subdirectory is created automatically. If multiple audio features share the same
`lazy_cache_dir`, each feature gets its own subdirectory named after the feature.

!!! note
    `lazy_cache_dir` controls the file cache for in-memory sources. The decoded memmap used by
    `lazy_cached` mode is placed next to the Parquet cache, not in `lazy_cache_dir`.

### When to Use `eager` Mode

Set `mode: eager` when:

- Your dataset is small enough to fit in memory and you want the fastest possible training start.
- You are running on a system without a persistent filesystem (e.g. some ephemeral cloud environments)
  and cannot write a cache.
- You are using a remote dataset backend that cannot deliver paths to local files.

```yaml
input_features:
  - name: audio
    type: audio
    preprocessing:
      mode: eager   # decode everything at preprocessing time
```

### Bare Tensor Inputs

If your dataset delivers bare `torch.Tensor` objects (shape `(channels, samples)` or `(samples,)`)
instead of dicts, Ludwig treats them the same as the in-memory dict case: tensors are written to WAV
files in `lazy_cache_dir` using the sample rate recorded in the feature metadata.

# Input Features

Audio files are transformed into one of the following types according to `type` under the `preprocessing` configuration.

- **`raw`**: Audio file is transformed into a float valued tensor of size `N x L x W` (where `N` is the size of the dataset and `L` corresponds to `audio_file_length_limit_in_s * sample_rate` and `W = 1`).
- **`stft`**: Audio is transformed to the `stft` magnitude. Audio file is transformed into a float valued tensor of size `N x L x W` (where `N` is the size of the dataset, `L` corresponds to `ceil(audio_file_length_limit_in_s * sample_rate - window_length_in_s * sample_rate + 1/ window_shift_in_s * sample_rate) + 1` and `W` corresponds to `num_fft_points / 2`).
- **`fbank`**: Audio file is transformed to FBANK features (also called log Mel-filter bank values). FBANK features are implemented according to their definition in the [HTK Book](http://www.inf.u-szeged.hu/~tothl/speech/htkbook.pdf): Raw Signal -> Preemphasis -> DC mean removal -> `stft` magnitude -> Power spectrum: `stft^2` -> mel-filter bank values: triangular filters equally spaced on a Mel-scale are applied -> log-compression: `log()`. Overall the audio file is transformed into a float valued tensor of size `N x L x W` with `N,L` being equal to the ones in `stft` and `W` being equal to `num_filter_bands`.
- **`stft_phase`**: The phase information for each stft bin is appended to the `stft` magnitude so that the audio file is transformed into a float valued tensor of size `N x L x 2W` with `N,L,W` being equal to the ones in `stft`.
- **`group_delay`**: Audio is transformed to group delay features according to Equation (23) in this [paper](https://www.ias.ac.in/article/fullyext/sadh/036/05/0745-0782). Group_delay features has the same tensor size as `stft`.

The encoder parameters specified at the feature level are:

- **`tied`** (default `null`): name of another input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

Example audio feature entry in the input features list:

```yaml
name: audio_column_name
type: audio
tied: null
encoder: 
    type: parallel_cnn
```

## Encoders

Audio feature encoders include all [Sequence Features](sequence_features.md#input-features) encoders as well as the
pretrained audio encoders described below.

Encoder type and encoder parameters can also be defined once and applied to all audio input features using the [Type-Global Encoder](../defaults.md#type-global-encoder) section.

### Wav2Vec2 Encoder

The Wav2Vec2 encoder (Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations", NeurIPS 2020) processes raw audio waveforms using self-supervised contrastive learning
over masked latent representations. It produces contextualized speech features suitable for speech
recognition, audio classification, and speaker identification.

Wav2Vec2 expects raw waveform input at 16kHz sample rate.

Default pretrained model: `facebook/wav2vec2-base`

{% set wav2vec2_encoder = get_encoder_schema("audio", "wav2vec2") %}
{{ render_yaml(wav2vec2_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(wav2vec2_encoder, exclude=["type"])) }}

### Whisper Encoder

The Whisper encoder (Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision",
ICML 2023) is the encoder portion of OpenAI's Whisper model, trained on 680,000 hours of multilingual
audio data. It excels at noisy and multilingual speech tasks.

Whisper expects log-mel spectrogram input (80 mel bins).

Default pretrained model: `openai/whisper-base`

{% set whisper_encoder = get_encoder_schema("audio", "whisper") %}
{{ render_yaml(whisper_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(whisper_encoder, exclude=["type"])) }}

### HuBERT Encoder

The HuBERT encoder (Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked
Prediction of Hidden Units", IEEE/ACM TASLP 2021) uses self-supervised masked prediction to learn
speech representations. It is particularly effective for speaker verification, emotion recognition,
and audio classification tasks.

HuBERT expects raw waveform input at 16kHz sample rate.

Default pretrained model: `facebook/hubert-base-ls960`

{% set hubert_encoder = get_encoder_schema("audio", "hubert") %}
{{ render_yaml(hubert_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(hubert_encoder, exclude=["type"])) }}

# Output Features

There are no audio decoders at the moment.

If this unlocks an interesting use case for your application, please file a GitHub Issue or ping the
[Community Discord](https://discord.gg/CBgdrGnZjy).
