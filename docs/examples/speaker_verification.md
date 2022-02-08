This example describes how to use Ludwig for a simple speaker verification task.
We assume to have the following data with label 0 corresponding to an audio file of an unauthorized voice and
label 1 corresponding to an audio file of an authorized voice.
The sample data looks as follows:

| audio_path                 | label |
| -------------------------- | ----- |
| audiodata/audio_000001.wav | 0     |
| audiodata/audio_000002.wav | 0     |
| audiodata/audio_000003.wav | 1     |
| audiodata/audio_000004.wav | 1     |

```
ludwig experiment \
--dataset speaker_verification.csv \
  --config_file config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: audio_path
        type: audio
        preprocessing:
            audio_file_length_limit_in_s: 7.0
            audio_feature:
                type: stft
                window_length_in_s: 0.04
                window_shift_in_s: 0.02
        encoder: cnnrnn

output_features:
    -
        name: label
        type: binary
```