{% from './macros/includes.md' import render_fields, render_yaml %}
{% set mv_details = "See [Missing Value Strategy](./input_features.md#missing-value-strategy) for details." %}
{% set norm_details = "See [Normalization](../combiner.md#normalization) for details." %}
{% set details = {"missing_value_strategy": mv_details, "fc_norm": norm_details} %}

# Preprocessing

Category features are transformed into integer valued vectors of size `n` (where `n` is the size of the dataset) and
added to the HDF5 with a key that reflects the name of column in the dataset.
Categories are mapped to integers by first collecting a dictionary of all unique category strings present in the column
of the dataset, ranking them descending by frequency and assigning a sequential integer ID from the most frequent to the
most rare (with 0 assigned to the special unknown placeholder token `<UNK>`).
The column name is added to the JSON file, with an associated dictionary containing

1. the mapping from integer to string (`idx2str`)
2. the mapping from string to id (`str2idx`)
3. the mapping from string to frequency (`str2freq`)
4. the size of the set of all tokens (`vocab_size`)
5. additional preprocessing information (by default how to fill missing values and what token to use to fill missing values)

{% set preprocessing = get_feature_preprocessing_schema("category") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing), details=details) }}

Preprocessing parameters can also be defined once and applied to all category input features using the [Type-Global Preprocessing](../defaults.md#type-global-preprocessing) section.

# Input Features

Category features have three encoders.
The `passthrough` encoder passes the raw integer values coming from the input placeholders to outputs of size `b x 1`.
The other two encoders map to either `dense` or `sparse` embeddings (one-hot encodings) and returned as outputs of size
`b x h`, where `b` is the batch size and `h` is the dimensionality of the embeddings.

The encoder parameters specified at the feature level are:

- **`tied`** (default `null`): name of another input feature to tie the weights of the encoder with. It needs to be the name of
a feature of the same type and with the same encoder parameters.

Example category feature entry in the input features list:

```yaml
name: category_column_name
type: category
tied: null
encoder: 
    type: dense
```

The available encoder parameters are:

- **`type`** (default `dense`): the possible values are `passthrough`, `dense`, `sparse`, `onehot`, `target` and `hash`.
`passthrough` outputs the raw integer values unaltered. `dense` randomly initializes a trainable embedding matrix,
`sparse` uses one-hot encoding, `onehot` produces a one-hot vector, `target` uses mean target encoding, and `hash`
uses feature hashing for fixed-memory encoding of high-cardinality categories.

Encoder type and encoder parameters can also be defined once and applied to all category input features using
the [Type-Global Encoder](../defaults.md#type-global-encoder) section.

## Encoders

### Dense Encoder

{% set encoder = get_encoder_schema("category", "dense") %}
{{ render_yaml(encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder, exclude=["type"]), details=details) }}

### Sparse Encoder

{% set encoder = get_encoder_schema("category", "sparse") %}
{{ render_yaml(encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(encoder, exclude=["type"]), details=details) }}

### OneHot Encoder

The `onehot` encoder produces a one-hot vector representation of the category. Each category is mapped to a
binary vector of size equal to the vocabulary size, with a single 1 at the position corresponding to the
category's index.

{% set onehot_encoder = get_encoder_schema("category", "onehot") %}
{{ render_yaml(onehot_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(onehot_encoder, exclude=["type"]), details=details) }}

### Target Encoder

The `target` encoder replaces each category with the mean of the target variable for that category
(also known as mean target encoding). This is effective for high-cardinality categorical features where
a standard embedding table would be very large. Ludwig handles target leakage internally using
cross-fitting on the training data.

{% set target_encoder = get_encoder_schema("category", "target") %}
{{ render_yaml(target_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(target_encoder, exclude=["type"]), details=details) }}

### Hash Encoder

The `hash` encoder uses feature hashing (the "hashing trick") to map categories to a fixed-size
embedding vector. This provides a constant memory footprint regardless of vocabulary size, handles
unseen categories gracefully, and works well for extremely high-cardinality features or streaming
data where the full vocabulary is not known in advance.

{% set hash_encoder = get_encoder_schema("category", "hash") %}
{{ render_yaml(hash_encoder, parent="encoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(hash_encoder, exclude=["type"]), details=details) }}

# Output Features

``` mermaid
graph LR
  A["Combiner\n Output"] --> B["Fully\n Connected\n Layers"];
  B --> C["Projection into\n Output Space"];
  C --> D["Softmax"];
  subgraph DEC["DECODER.."]
  B
  C
  D
  end
```
{ data-search-exclude }

Category features can be used when a multi-class classification needs to be performed.
There is only one decoder available for category features: a (potentially empty) stack of fully connected layers,
followed by a projection into a vector of size of the number of available classes, followed by a softmax.

Example category output feature using default parameters:

```yaml
name: category_column_name
type: category
reduce_input: sum
dependencies: []
calibration: false
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: null
    class_similarities: null
    class_similarities_temperature: 0
decoder:
    type: classifier
```

Parameters:

- **`reduce_input`** (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order
tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`,
`max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- **`calibration`** (default `false`): if true, performs calibration by temperature scaling after training is complete.
Calibration uses the validation set to find a scale factor (temperature) which is multiplied with the logits to shift
output probabilities closer to true likelihoods.
- **`dependencies`** (default `[]`): the output features this one is dependent on. For a detailed explanation refer to
[Output Features Dependencies](output_features.md#output-feature-dependencies).
- **`reduce_dependencies`** (default `sum`): defines how to reduce the output of a dependent feature that is not a vector,
but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available
values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last
vector of the first dimension).
- **`loss`** (default `{type: softmax_cross_entropy}`): is a dictionary containing a loss `type`. `softmax_cross_entropy` is
the only supported loss type for category output features. See [Loss](#loss) for details.
- **`top_k`** (default `3`): determines the parameter `k`, the number of categories to consider when computing the `top_k`
measure. It computes accuracy but considering as a match if the true category appears in the first `k` predicted
categories ranked by decoder's confidence.
- **`decoder`** (default: `{"type": "classifier"}`): Decoder for the desired task. Options: `classifier`, `mlp_classifier`. See [Decoder](#decoders) for details.

Decoder type and decoder parameters can also be defined once and applied to all category output features using the [Type-Global Decoder](../defaults.md#type-global-decoder) section.

## Decoders

### Classifier

{% set decoder = get_decoder_schema("category", "classifier") %}
{{ render_yaml(decoder, parent="decoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(decoder, exclude=["type"]), details=details) }}

### MLP Classifier

`mlp_classifier` adds an explicit stack of hidden layers between the combiner output and the
final projection. It is useful when the combiner itself is shallow (for example the `concat`
combiner over tabular features) and the classification head needs more non-linear capacity.

```yaml
decoder:
  type: mlp_classifier
  num_fc_layers: 2
  output_size: 512
  activation: relu
  dropout: 0.2
```

Setting `num_fc_layers: 0` is equivalent to the plain `classifier`.

{% set decoder = get_decoder_schema("category", "mlp_classifier") %}
{{ render_yaml(decoder, parent="decoder") }}

Parameters:

{{ render_fields(schema_class_to_fields(decoder, exclude=["type"]), details=details) }}

### Probability calibration (temperature scaling)

Both `classifier` and `mlp_classifier` expose a `calibration` parameter. Set it to
`temperature_scaling` to learn a single scalar temperature on the validation set that is
applied as `logits / T` ([Guo et al., ICML 2017](https://arxiv.org/abs/1706.04599)).
Temperature scaling never changes the argmax prediction; it only reshapes the probability
distribution so that predicted confidences track empirical accuracy more closely.

```yaml
decoder:
  type: classifier
  calibration: temperature_scaling
```

Temperature scaling is run automatically at the end of training using the validation split.
The learned temperature is serialized with the model and applied at inference.

### Monte Carlo dropout uncertainty

Both decoders also expose `mc_dropout_samples` ([Gal & Ghahramani, ICML 2016](https://arxiv.org/abs/1506.02142)).
When set to a positive integer, Ludwig runs the decoder that many times at inference *with
dropout enabled* and returns:

- the mean predicted probabilities,
- an additional `uncertainty` tensor that captures the variance across samples.

```yaml
decoder:
  type: mlp_classifier
  num_fc_layers: 2
  dropout: 0.2
  mc_dropout_samples: 20
```

MC dropout requires `dropout > 0` in the decoder; otherwise the repeated forward passes are
identical and the uncertainty estimate is always zero.

## Loss

### Softmax Cross Entropy

{% set loss = get_loss_schema("softmax_cross_entropy") %}
{{ render_yaml(loss, parent="loss") }}

Parameters:

{{ render_fields(schema_class_to_fields(loss, exclude=["type"]), details=details) }}

Loss and loss related parameters can also be defined once and applied to all category output features using the [Type-Global Loss](../defaults.md#type-global-loss) section.

## Metrics

The measures that are calculated every epoch and are available for category features are `accuracy`, `hits_at_k`
(computes accuracy considering as a match if the true category appears in the first `k` predicted categories ranked by
decoder's confidence) and the `loss` itself.
You can set either of them as `validation_metric` in the `training` section of the configuration if you set the
`validation_field` to be the name of a category feature.

---

# Category Distribution Output Feature

`category_distribution` is a specialised output feature type for datasets where each target row is a **soft probability
distribution over categories** rather than a single hard label.  Typical use cases include:

- Label smoothing targets generated from an ensemble or teacher model
- Crowd-sourced annotations summarised as empirical agreement rates
- Reading comprehension tasks with partial-credit labels

The feature type is `category_distribution`.  Because the targets are probability vectors, the column values must be
parseable as JSON arrays of floats whose length equals the number of classes.

## Configuration

```yaml
output_features:
  - name: label_distribution
    type: category_distribution
    vocab: [class_a, class_b, class_c]
    preprocessing:
      missing_value_strategy: drop_row
    decoder:
      type: classifier
    loss:
      type: softmax_cross_entropy
```

### Required parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `vocab` | list[str] | Ordered list of class names.  **Required** — `category_distribution` has no way to infer the vocabulary from the soft-label vectors. |

### Preprocessing

{% set preprocessing = get_feature_preprocessing_schema("category_distribution_output") %}
{{ render_yaml(preprocessing, parent="preprocessing") }}

Parameters:

{{ render_fields(schema_class_to_fields(preprocessing), details=details) }}

### Decoders

`category_distribution` reuses the same `classifier` and `mlp_classifier` decoders as the standard `category` output
feature.  See [Decoders](#decoders) above for parameter details.

### Loss

`category_distribution` targets are treated as probability distributions, so the recommended loss is
`softmax_cross_entropy` with label smoothing disabled (smoothing is already encoded in the targets).

### Metrics

The same metrics as `category` are reported: `accuracy` (argmax of the predicted distribution vs. argmax of the target
distribution), `hits_at_k`, and `loss`.
