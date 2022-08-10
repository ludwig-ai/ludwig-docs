## Audio Features Preprocessing

Example of a preprocessing specification (assuming the audio files have a sample rate of 16000):

```yaml
name: audio_path
type: audio
preprocessing:
  audio_file_length_limit_in_s: 7.5
  audio_feature:
    type: stft
    window_length_in_s: 0.04
    window_shift_in_s: 0.02
    num_fft_points: 800
    window_type: boxcar
```

Ludwig supports reading audio files using PyTorch's [Torchaudio](https://pytorch.org/audio/stable/index.html) library. This library supports `WAV`, `AMB`, `MP3`, `FLAC`, `OGG/VORBIS`, `OPUS`, `SPHERE`, and `AMR-NB` formats.

Preprocessing parameters:

- `audio_file_length_limit_in_s`: (default `7.5`): float value that defines the maximum limit of the audio file in seconds. All files longer than this limit are cut off. All files shorter than this limit are padded with `padding_value`
- `missing_value_strategy` (default: `backfill`): what strategy to follow when there's a missing value in an audio column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `in_memory` (default `true`): defines whether an audio dataset will reside in memory during the training process or will be dynamically fetched from disk (useful for large datasets). In the latter case a training batch of input audio files will be fetched from disk each training iteration. At the moment only `in_memory` = true is supported.
- `padding_value`: (default 0): float value that is used for padding.
- `norm`: (default `null`) the normalization method that can be used for the input data. Supported methods: `null` (data is not normalized), `per_file` (z-norm is applied on a “per file” level)
- `audio_feature`: (default `{ type: raw }`) dictionary that takes as input the audio feature `type` as well as additional parameters if `type != raw`. The following parameters can/should be defined in the dictionary:
  - `type` (default `raw`): defines the type of audio features to be used. Supported types at the moment are `raw`, `stft`, `stft_phase`, `group_delay`. For more detail, check [Audio Input Features and Encoders](#audio-input-features-and-encoders).
  - `window_length_in_s`: defines the window length used for the short time Fourier transformation (only needed if `type != raw`).
  - `window_shift_in_s`: defines the window shift used for the short time Fourier transformation (also called hop_length) (only needed if `type != raw`).
  - `num_fft_points`: (default `window_length_in_s * sample_rate` of audio file) defines the number of fft points used for the short time Fourier transformation. If `num_fft_points > window_length_in_s * sample_rate`, then the signal is zero-padded at the end. `num_fft_points` has to be `>= window_length_in_s * sample_rate` (only needed if `type != raw`).
  - `window_type`: (default `hamming`): defines the type window the signal is weighted before the short time Fourier transformation. All windows provided by [scipy’s window function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html) can be used (only needed if `type != raw`).
  - `num_filter_bands`: defines the number of filters used in the filterbank (only needed if `type == fbank`).

Preprocessing parameters can also be defined once and applied to all audio input features using the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

## Audio Input Features and Encoders

Audio files are transformed into one of the following types according to `type` in `audio_feature`'s `preprocessing` configuration.

- `raw`: audio file is transformed into a float valued tensor of size `N x L x W` (where `N` is the size of the dataset and `L` corresponds to `audio_file_length_limit_in_s * sample_rate` and `W = 1`).
- `stft`: audio is transformed to the `stft` magnitude. Audio file is transformed into a float valued tensor of size `N x L x W` (where `N` is the size of the dataset, `L` corresponds to `ceil(audio_file_length_limit_in_s * sample_rate - window_length_in_s * sample_rate + 1/ window_shift_in_s * sample_rate) + 1` and `W` corresponds to `num_fft_points / 2`).
- `fbank`: audio file is transformed to FBANK features (also called log Mel-filter bank values). FBANK features are implemented according to their definition in the [HTK Book](http://www.inf.u-szeged.hu/~tothl/speech/htkbook.pdf): Raw Signal -> Preemphasis -> DC mean removal -> `stft` magnitude -> Power spectrum: `stft^2` -> mel-filter bank values: triangular filters equally spaced on a Mel-scale are applied -> log-compression: `log()`. Overall the audio file is transformed into a float valued tensor of size `N x L x W` with `N,L` being equal to the ones in `stft` and `W` being equal to `num_filter_bands`.
- `stft_phase`: the phase information for each stft bin is appended to the `stft` magnitude so that the audio file is transformed into a float valued tensor of size `N x L x 2W` with `N,L,W` being equal to the ones in `stft`.
- `group_delay`: audio is transformed to group delay features according to Equation (23) in this [paper](https://www.ias.ac.in/article/fullyext/sadh/036/05/0745-0782). Group_delay features has the same tensor size as `stft`.

Audio feature encoders are the same as for [Sequence Features](../sequence_features#sequence-input-features-and-encoders).

Encoder type and encoder parameters can also be defined once and applied to all audio input features using the [Type-Global Encoder](../defaults.md#type-global-encoder) section.

## Audio Output Features and Decoders

There are no audio decoders at the moment.

If this would unlock an interesting use case for your application, please file a GitHub Issue or ping the
[Ludwig Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ).
