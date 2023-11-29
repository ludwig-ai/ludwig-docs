This is a complete example of training an spoken digit speech recognition model on the "MNIST dataset of speech recognition".

## Download the free spoken digit dataset

```
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
mkdir speech_recog_digit_data
cp -r free-spoken-digit-dataset/recordings speech_recog_digit_data
cd speech_recog_digit_data
```

## Create an CSV dataset

```
echo "audio_path","label" >> "spoken_digit.csv"
cd "recordings"
ls | while read -r file_name; do
   audio_path=$(readlink -m "${file_name}")
   label=$(echo ${file_name} | cut -c1)
   echo "${audio_path},${label}" >> "../spoken_digit.csv"
done
cd "../"
```

Now you should have `spoken_digit.csv` containing 2000 examples having the following format

| audio_path                                              | label |
| ------------------------------------------------------- | ----- |
| .../speech_recog_digit_data/recordings/0_jackson_0.wav  | 0     |
| .../speech_recog_digit_data/recordings/0_jackson_10.wav | 0     |
| .../speech_recog_digit_data/recordings/0_jackson_11.wav | 0     |
| ...                                                     | ...   |
| .../speech_recog_digit_data/recordings/1_jackson_0.wav  | 1     |

## Train a model

From the directory where you have virtual environment with ludwig installed:

```
ludwig experiment \
  --dataset <PATH_TO_SPOKEN_DIGIT_CSV> \
  --config config_file.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: audio_path
        type: audio
        encoder: 
            type: stacked_cnn
            reduce_output: concat
            conv_layers:
                -
                    num_filters: 16
                    filter_size: 6
                    pool_size: 4
                    pool_stride: 4
                    dropout: 0.4
                -
                    num_filters: 32
                    filter_size: 3
                    pool_size: 2
                    pool_stride: 2
                    dropout: 0.4
            fc_layers:
                -
                    output_size: 64
                    dropout: 0.4
        preprocessing:
            audio_feature:
                type: fbank
                window_length_in_s: 0.025
                window_shift_in_s: 0.01
                num_filter_bands: 80
            audio_file_length_limit_in_s: 1.0
            norm: per_file

output_features:
    -
        name: label
        type: category

trainer:
    early_stop: 10
```
