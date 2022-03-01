## Bag Features Preprocessing

Bag features are expected to be provided as a string of elements separated by whitespace, e.g. "elem5 elem0 elem5 elem1".
Bags are similar to [set features](../set_features), the only difference being that elements may appear multiple
times. The bag feature encoder outputs a matrix, similar to a set encoder, except each element of the matrix is a float
value representing the frequency of the respective element in the bag. Embeddings are aggregated by summation, weighted
by the frequency of each element.

## Bag Input Features and Encoders

Bag features have only one encoder type available.

### Embed Weighted Encoder

The embed weighted encoder first transforms the element frequency vector to sparse integer lists, which are then mapped
to either dense or sparse embeddings (one-hot encodings). Lastly, embeddings are aggregated as a weighted sum where each
embedding is multiplied by its respective element's frequency.
Inputs are of size `b` while outputs are of size `b x h` where `b` is the batch size and `h` is the dimensionality of
the embeddings.

The parameters are the same used for [set input features](../set_features#set-input-features-and-encoders) except for
`reduce_output` which should not be used because the weighted sum already acts as a reducer.
g

```
+---+
|0.0|          +-----+
|1.0|   +-+    |emb 0|   +-----------+
|1.0|   |0|    +-----+   |Weighted   |
|0.0+--->1+---->emb 1+--->Sum        +->
|0.0|   |5|    +-----+   |Operation  |
|2.0|   +-+    |emb 5|   +-----------+
|0.0|          +-----+
+---+
```

Example bag feature entry in the input features list:

```yaml
name: bag_column_name
type: bag
representation: dense
tied_weights: null
```

## Bag Output Features and Decoders

There is no bag decoder available yet.

## Bag Features Metrics

As there is no decoder there is also no metric available yet for bag features.
