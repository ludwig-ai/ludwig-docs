## Time Series Features Preprocessing

Timeseries features are handled as sequence features, with the only difference being that the matrix in the HDF5
preprocessing file uses floats instead of integers.

Since data is continuous, the JSON file, which typically stores vocabulary mappings, isn't needed.

## Time Series Input Features and Encoders

Time series encoders are the same as for [Sequence Features](../sequence_features#sequence-input-features-and-encoders), with one exception:

Time series features don't have an embedding layer at the beginning, so the `b x s` placeholders (where `b` is the batch
size and `s` is the sequence length) are directly mapped to a `b x s x 1` tensor and then passed to the different
sequential encoders.

## Time Series Output Features and Decoders

There are no time series decoders at the moment.

If this would unlock an interesting use case for your application, please file a GitHub Issue or ping the
[Ludwig Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ).
