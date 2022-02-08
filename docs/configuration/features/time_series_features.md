## Time Series Features Preprocessing

Timeseries features are treated in the same way of sequence features, with the only difference being that the matrix in the HDF5 file does not have integer values, but float values.
Moreover, there is no need for any mapping in the JSON file.

## Time Series Input Features and Encoders

The encoders are the same used for the [Sequence Features](#sequence-input-features-and-encoders).
The only difference is that time series features don't have an embedding layer at the beginning, so the `b x s` placeholders (where `b` is the batch size and `s` is the sequence length) are directly mapped to a `b x s x 1` tensor and then passed to the different sequential encoders.

## Time Series Output Features and Decoders

There are no time series decoders at the moment (WIP), so time series cannot be used as output features.

## Time Series Features Measures

As no time series decoders are available at the moment, there are also no time series measures.
