Date features are like `2023-06-25 15:00:00`, `2023-06-25`, `6-25-2023`, or `6/25/2023`.

## Date Features Preprocessing

```yaml
name: date_feature_name
type: date
preprocessing:
  missing_value_strategy: fill_with_const
  fill_value: ''
  datetime_format: "%d %b %Y"
```

Ludwig will try to infer the date format automatically, but a specific format can be provided. The date string spec is
the same as the one described in python's [datetime](https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior).

Here are some preprocessing parameters.

- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter). The special value `backfill` replaces the missing values with the next valid value.
- `fill_value` (default `""`): the value to replace missing values with when `missing_value_strategy` is `fill_with_const`. This can be a datetime string. If empty, the current datetime will be used.
- `datetime_format` (default `null`): this parameter can either be a datetime format string, or `null`, in which case the datetime format will be inferred automatically.

## Date Input Features and Encoders

Input date features are transformed into a int tensors of size `N x 9` (where `N` is the size of the dataset and the 9 dimensions contain year, month, day, weekday, yearday, hour, minute, second, and second of day).

For example, the date `2022-06-25 09:30:59` would be deconstructed into:

```python
[
  2022, # Year
  6,  # June
  25, # 25th day of the month
  5,  # Weekday: Saturday
  176,  # 176th day of the year
  9,  # Hour
  30, # Minute
  59, # Seconds
  34259,  # 34259th second of the day
]
```

Currently there are two encoders supported for dates: `DateEmbed` (default) and `DateWave`. The encoder can be set by
specifying `embed` or `wave` in the feature's `encoder` parameter in the input feature's configuration.

```yaml
name: date_feature_name
type: date
encoder: embed
```

### Embed Encoder

```yaml
name: date_column_name
type: date
encoder: embed
embedding_size: 10
embeddings_on_cpu: false
dropout: false
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
activation: relu
dropout: 0
```

This encoder passes the year through a fully connected layer of one neuron and embeds all other elements for the date,
concatenates them and passes the concatenated representation through fully connected layers.

It takes the following optional parameters:

- `embedding_size` (default `10`): it is the maximum embedding size adopted.
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `dropout` (default `0`): dropout rate.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `0`): This is the number of stacked fully connected layers.
- `output_size` (default `10`): if a `output_size` is not already specified in `fc_layers` this is the default `output_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Torch's documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) or for `layer` see [Torch's documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

### Wave Encoder

```yaml
name: date_column_name
type: date
encoder: wave
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
norm: null
norm_params: null
activation: relu
dropout: 0
```

This encoder passes the year through a fully connected layer of one neuron and represents all other elements for the date by taking the cosine of their value with a different period (12 for months, 31 for days, etc.), concatenates them and passes the concatenated representation through fully connected layers.

It takes the following parameters:

- `fc_layers` (default `null`): a list of dictionaries containing the parameters of all the fully connected
layers. The length of the list determines the number of stacked fully connected layers and the content of each
dictionary determines the parameters for a specific layer. The available parameters for each layer are: `activation`,
`dropout`, `norm`, `norm_params`, `output_size`, `use_bias`, `bias_initializer` and `weights_initializer`. If any of
those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used
instead.
- `num_fc_layers` (default `0`): This is the number of stacked fully connected layers.
- `fc_size` (default `10`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Torch's documentation on batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) or for `layer` see [Torch's documentation on layer normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

## Date Output Features and Decoders

There is currently no support for date as an output feature. Consider using the [`TEXT` type](../../features/text_features).
