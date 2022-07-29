This example illustrates univariate timeseries forecasting using historical temperature data for Los Angeles.

Dowload and unpack historical hourly weather data available on Kaggle
<https://www.kaggle.com/selfishgene/historical-hourly-weather-data>

Run the following python script to prepare the training dataset:

```
import pandas as pd
from ludwig.utils.data_utils import add_sequence_feature_column

df = pd.read_csv(
    '<PATH_TO_FILE>/temperature.csv',
    usecols=['Los Angeles']
).rename(
    columns={"Los Angeles": "temperature"}
).fillna(method='backfill').fillna(method='ffill')

# normalize
df.temperature = ((df.temperature-df.temperature.mean()) /
                  df.temperature.std())

train_size = int(0.6 * len(df))
vali_size = int(0.2 * len(df))

# train, validation, test split
df['split'] = 0
df.loc[
    (
        (df.index.values >= train_size) &
        (df.index.values < train_size + vali_size)
    ),
    ('split')
] = 1
df.loc[
    df.index.values >= train_size + vali_size,
    ('split')
] = 2

# prepare timeseries input feature colum
# (here we are using 20 preceding values to predict the target)
add_sequence_feature_column(df, 'temperature', 20)
df.to_csv('<PATH_TO_FILE>/temperature_la.csv')
```

```
ludwig experiment \
--dataset <PATH_TO_FILE>/temperature_la.csv \
  --config config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: temperature_feature
        type: timeseries
        encoder: 
            type: rnn
            embedding_size: 32
            state_size: 32

output_features:
    -
        name: temperature
        type: number
```
