{% from './macros/includes.md' import render_fields, render_yaml %}
{% set mv_details = "See [Missing Value Strategy](./input_features.md#missing-value-strategy) for details." %}
{% set type = "See explanations for each type [here](audio_features.md#audio-input-features-and-encoders)." %}
{% set details = {"missing_value_strategy": mv_details, "type": type} %}

# Preprocessing

Example of a preprocessing specification (assuming the audio files have a sample rate of 16000):

{% set preprocessing = get_feature_preprocessing_schema("audio") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

Ludwig supports reading audio files using PyTorch's [Torchaudio](https://pytorch.org/audio/stable/index.html) library. This library supports `WAV`, `AMB`, `MP3`, `FLAC`, `OGG/VORBIS`, `OPUS`, `SPHERE`, and `AMR-NB` formats.

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing), details=details) }}

Preprocessing parameters can also be defined once and applied to all audio input features using the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

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

Audio feature encoders are the same as for [Sequence Features](../sequence_features#sequence-input-features-and-encoders).

Encoder type and encoder parameters can also be defined once and applied to all audio input features using the [Type-Global Encoder](../defaults.md#type-global-encoder) section.

# Output Features

There are no audio decoders at the moment.

If this unlocks an interesting use case for your application, please file a GitHub Issue or ping the
[Community Discord](https://discord.gg/CBgdrGnZjy).
