The configuration is the core of Ludwig. 
It is a dictionary that contains all the information needed to build and train a Ludwig model.
It mixes ease of use, by means of reasonable defaults, with flexibility, by means of detailed control over the parameters of your model.
It is provided to both `experiment` and `train` commands either as a string (`--config`) or as a file (`--config_file`).
The string or the content of the file will be parsed by PyYAML into a dictionary in memory, so any style of YAML accepted by the parser is considered to be valid, so both multiline and oneline formats are accepted.
For instance a list of dictionaries can be written both as:

```yaml
mylist: [{name: item1, score: 2}, {name: item2, score: 1}, {name: item3, score: 4}]
```

or as:

```yaml
mylist:
    -
        name: item1
        score: 2
    -
        name: item2
        score: 1
    -
        name: item3
        score: 4
```

The structure of the configuration file is a dictionary with five keys:

```yaml
input_features: []
combiner: {}
output_features: []
training: {}
preprocessing: {}
```

Only `input_features` and `output_features` are required, the other three fields 
have default values, but you are free to modify them.

Input features
--------------

The `input_features` list contains a list of dictionaries, each of them containing two required fields `name` and `type`.
`name` is the name of the feature and is the same name of the column of the dataset input file, `type` is one of the supported datatypes.
Input features may have different ways to be encoded and the parameter to decide it is `encoder`.

All the other parameters you specify in an input feature will be passed as parameters to the function that build the encoder, and each encoder can have different parameters.

For instance a `sequence` feature can be encoded by a `stacked_cnn` or by and `rnn`, but only the `stacked_cnn` will accept the parameter `num_filters` while only the `rnn` will accept the parameter `bidirectional`.

A list of all the encoders available for all the datatypes alongside with the description of all parameters will be provided in the datatype-specific sections.
Some datatypes have only one type of encoder, so you are not required to specify it.

The role of the encoders is to map inputs into tensors, usually vectors in the case of datatype without a temporal / sequential aspect, matrices in case there is a temporal / sequential aspect or higher rank tensors in case there is a spatial or a spatio-temporal aspect to the input data.

Different configurations of the same encoder may return a tensor with different rank, for instance a sequential encoder may return a vector of size `h` that is either the final vector of a sequence or the result of pooling over the sequence length, or it can return a matrix of size `l x h` where `l` is the length of the sequence and `h` is the hidden dimension if you specify the pooling reduce operation (`reduce_output`) to be `null`.  For the sake of simplicity you can imagine the output to be a vector in most of the cases, but there is a `reduce_output` parameter one can specify to change the default behavior.

An additional feature that Ludwig provides is the option to have tied weights between different encoders.
For instance if my model takes two sentences as input and return the probability of their entailment, I may want to encode both sentences with the same encoder.
The way to do it is by specifying the `tied-weights` parameter of the second feature you define to be the name of the first feature you defined.

```yaml
input_features:
    -
        name: sentence1
        type: text
    -
        name: sentence2
        type: text
        tied_weights: sentence1
```

If you specify a name of an input feature that has not been defined yet, it will result in an error.
Also, in order to be able to have tied weights, all encoder parameters have to be identical between the two input features.

Combiner
--------

Combiners are part of the model that take all the outputs of the different input features and combine them in a single representation that is passed to the outputs.
You can specify which one to use in the `combiner` section of the configuration.
Different combiners implement different combination logic, but the default one `concat` just concatenates all outputs of input feature encoders and optionally passes the concatenation through fully connected layers, with the output of the last layer being forwarded to the outputs decoders.

```
+-----------+
|Input      |
|Feature 1  +-+
+-----------+ |            +---------+
+-----------+ | +------+   |Fully    |
|...        +--->Concat+--->Connected+->
+-----------+ | +------+   |Layers   |
+-----------+ |            +---------+
|Input      +-+
|Feature N  |
+-----------+
```

For the sake of simplicity you can imagine the both inputs and outputs are vectors in most of the cases, but there are `reduce_input` and `reduce_output` parameters to specify to change the default behavior.

Output Features
---------------

The `output_features` list has the same structure of the `input_features` list: it is a list of dictionaries containing a `name` and a `type`.
They represent outputs / targets that you want your model to predict.
In most machine learning tasks you want to predict only one target variable, but in Ludwig you are allowed to specify as many outputs as you want and they are going to be optimized in a multi-task fashion, using a weighted sum of their losses as a combined loss to optimize.

Instead of having `encoders`, output features have `decoders`, but most of them have only one decoder so you don't have to specify it.

Decoders take the output of the combiner as input, process it further, for instance passing it through fully connected layers, and finally predict values and compute a loss and some measures (depending on the datatype different losses and measures apply).

Decoders have additional parameters, in particular `loss` that allows you to specify a different loss to optimize for this specific decoder, for instance numerical features support both `mean_squared_error` and `mean_absolute_error` as losses.
Details about the available decoders and losses alongside with the description of all parameters will be provided in the datatype-specific sections.

For the sake of simplicity you can imagine the input coming from the combiner to be a vector in most of the cases, but there is a `reduce_input` parameter one can specify to change the default behavior.

### Multi-task Learning

As Ludwig allows for multiple output features to be specified and each output feature can be seen as a task the model is learning to perform, by consequence Ludwig supports Multi-task learning natively.
When multiple output features are specified, the loss that is optimized is a weighted sum of the losses of each individual output feature.
By default each loss weight is `1`, but it can be changed by specifying a value for the `weight` parameter in the `loss` section of each output feature definition.

For example, given a `category` feature `A` and `numerical` feature `B`, in order to optimize the loss `loss_total = 1.5 * loss_A + 0.8 + loss_B` the `output_feature` section of the configuration should look like:

```yaml
output_features:
    -
        name: A
        type: category
        loss:
          weight: 1.5
    -
        name: A
        type: numerical
        loss:
          weight: 0.8
```

### Output Features Dependencies

An additional feature that Ludwig provides is the concept of dependency between `output_features`.  You can specify a list of output features as dependencies when you write the dictionary of a specific feature.
At model building time Ludwig checks that no cyclic dependency exists.
If you do so Ludwig will concatenate all the final representations before the prediction of those output features to the original input of the decoder.
The reason is that if different output features have a causal dependency, knowing which prediction has been made for one can help making the prediction of the other.

For instance if two output features are one coarse grained category and one fine-grained category that are in a hierarchical structure with each other, knowing the prediction made for coarse grained restricts the possible categories to predict for the fine-grained.
In this case the following configuration structure can be used:

```yaml
output_features:
    -
        name: coarse_class
        type: category
        num_fc_layers: 2
        fc_size: 64
    -
        name: fine_class
        type: category
        dependencies:
            - coarse_class
        num_fc_layers: 1
        fc_size: 64
```

Assuming the input coming from the combiner has hidden dimension `h` 128, there are two fully connected layers that return a vector with hidden size 64 at the end of the `coarse_class` decoder (that vector will be used for the final layer before projecting in the output `coarse_class` space).  In the decoder of `fine_class`, the 64 dimensional vector of `coarse_class` will be concatenated to the combiner output vector, making a vector of hidden size 192 that will be passed through a fully connected layer and the 64 dimensional output will be used for the final layer before projecting in the output class space of the `fine_class`.

Training
--------

The `training` section of the configuration lets you specify some parameters of the training process, like for instance the number of epochs or the learning rate.

These are the available training parameters:

- `batch_size` (default `128`): size of the batch used for training the model.
- `eval_batch_size` (default `0`): size of the batch used for evaluating the model. If it is `0`, the same value of `batch_size` is used. This is usefult to speedup evaluation with a much bigger batch size than training, if enough memory is available, or to decrease the batch size when `sampled_softmax_cross_entropy` is used as loss for sequential and categorical features with big vocabulary sizes (evaluation needs to be performed on the full vocabulary, so a much smaller batch size may be needed to fit the activation tensors in memory).
- `epochs` (default `100`): number of epochs the training process will run for.
- `early_stop` (default `5`): if there's a validation set, number of epochs of patience without an improvement on the validation measure before the training is stopped.
- `optimizer` (default `{type: adam, beta1: 0.9, beta2: 0.999, epsilon: 1e-08}`): which optimizer to use with the relative parameters. The available optimizers are: `sgd` (or `stochastic_gradient_descent`, `gd`, `gradient_descent`, they are all the same), `adam`, `adadelta`, `adagrad`, `adamax`, `ftrl`, `nadam`,
`rmsprop`. To know their parameters check [TensorFlow's optimizer documentation](https://www.tensorflow.org/api_docs/python/tf/train).
- `learning_rate` (default `0.001`): the learning rate to use.
- `decay` (default `false`): if to use exponential decay of the learning rate or not.
- `decay_rate` (default `0.96`): the rate of the exponential learning rate decay.
- `decay_steps` (default `10000`): the number of steps of the exponential learning rate decay.
- `staircase` (default `false`): decays the learning rate at discrete intervals.
- `regularization_lambda` (default `0`): the lambda parameter used for adding a l2 regularization loss to the overall loss.
- `reduce_learning_rate_on_plateau` (default `0`): if there's a validation set, how many times to reduce the learning rate when a plateau of validation measure is reached.
- `reduce_learning_rate_on_plateau_patience` (default `5`): if there's a validation set, number of epochs of patience without an improvement on the validation measure before reducing the learning rate.
- `reduce_learning_rate_on_plateau_rate` (default `0.5`): if there's a validation set, the reduction rate of the learning rate.
- `increase_batch_size_on_plateau` (default `0`): if there's a validation set, how many times to increase the batch size when a plateau of validation measure is reached.
- `increase_batch_size_on_plateau_patience` (default `5`): if there's a validation set, number of epochs of patience without an improvement on the validation measure before increasing the learning rate.
- `increase_batch_size_on_plateau_rate` (default `2`): if there's a validation set, the increase rate of the batch size.
- `increase_batch_size_on_plateau_max` (default `512`): if there's a validation set, the maximum value of batch size.
- `validation_field` (default `combined`): when there is more than one output feature, which one to use for computing if there was an improvement on validation. The measure to use to determine if there was an improvement can be set with the `validation_measure` parameter. Different datatypes have different available measures, refer to the datatype-specific section for more details. `combined` indicates the use the combination of all features. For instance the combination of `combined` and `loss` as measure uses a decrease in the combined loss of all output features to check for improvement on validation, while `combined` and `accuracy` considers on how many datapoints the predictions for all output features were correct (but consider that for some features, for instance `numeric` there is no accuracy measure, so you should use `accuracy` only if all your output features have an accuracy measure).
- `validation_metric:` (default `loss`): the metric to use to determine if there was an improvement. The metric is considered for the output feature specified in `validation_field`. Different datatypes have different available metrics, refer to the datatype-specific section for more details.
- `bucketing_field` (default `null`): when not `null`, when creating batches, instead of shuffling randomly, the length along the last dimension of the matrix of the specified input feature is used for bucketing datapoints and then randomly shuffled datapoints from the same bin are sampled. Padding is trimmed to the longest datapoint in the batch. The specified feature should be either a `sequence` or `text` feature and the encoder encoding it has to be `rnn`. When used, bucketing improves speed of `rnn` encoding up to 1.5x, depending on the length distribution of the inputs.
- `learning_rate_warmup_epochs` (default `1`): It's the number or training epochs where learning rate warmup will be used. It is calculated as described in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677). In the paper the authors suggest `6` epochs of warmup, that parameter is suggested for large datasets and big batches.

### Optimizers details

The available optimizers wrap the ones available in TensorFlow.
For details about the parameters pleease refer to the [TensorFlow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers).

The `learning_rate` parameter the optimizer will use come from the `training` section.
Other optimizer specific parameters, shown with their Ludwig default settings, follow:

- `sgd` (or `stochastic_gradient_descent`, `gd`, `gradient_descent`)
```
'momentum': 0.0,
'nesterov': false
```

- `adam`
```
'beta_1': 0.9,
'beta_2': 0.999,
'epsilon': 1e-08
```

- `adadelta`
```
'rho': 0.95,
'epsilon': 1e-08
```

- `adagrad`
```
'initial_accumulator_value': 0.1,
'epsilon': 1e-07
```

- `adamax`
```
'beta_1': 0.9, 
'beta_2': 0.999, 
'epsilon': 1e-07
```

- `ftrl`
``` 
'learning_rate_power': -0.5, 
'initial_accumulator_value': 0.1,
'l1_regularization_strength': 0.0, 
'l2_regularization_strength': 0.0,
```

- `nadam`,
```
'beta_1': 0.9, 
'beta_2': 0.999, 
'epsilon': 1e-07
```

- `rmsprop`
``` 
'decay': 0.9,
'momentum': 0.0,
'epsilon': 1e-10,
'centered': false
```


Preprocessing
-------------

The `preprocessing` section of the configuration makes it possible to specify datatype specific parameters to perform data preprocessing.
The preprocessing dictionary contains one key of each datatype, but you have to specify only the ones that apply to your case, the other ones will be kept as defaults.
Moreover, the preprocessing dictionary contains parameters related to how to split the data that are not feature specific.

- `force_split` (default `false`): if `true` the `split` column in the dataset file is ignored and the dataset is randomly split. If `false` the `split` column is used if available.
- `split_probabilities` (default `[0.7, 0.1, 0.2]`): the proportion of the dataset data to end up in training, validation and test, respectively. The three values have to sum up to one.
- `stratify` (default `null`): if `null` the split is random, otherwise you can specify the name of a `category` feature and the split will be stratified on that feature.

Example preprocessing dictionary (showing default values):

```yaml
preprocessing:
    force_split: false
    split_probabilities: [0.7, 0.1, 0.2]
    stratify: null
    category: {...}
    sequence: {...}
    text: {...}
    ...
```

The details about the preprocessing parameters that each datatype accepts will be provided in the datatype-specific sections.

It is important to point out that different features with the same datatype may require different preprocessing.
For instance a document classification model may have two text input features, one for the title of the document and one for the body.

As the length of the title is much shorter than the length of the body, the parameter `word_length_limit` should be set to 10 for the title and 2000 for the body, but both of them share the same parameter `most_common_words` with value 10000.

The way to do this is adding a `preprocessing` key inside the title `input_feature` dictionary and one in the `body` input feature dictionary containing the desired parameter and value.
The configuration will look like:

```yaml
preprocessing:
    text:
        most_common_word: 10000
input_features:
    -
        name: title
        type: text
        preprocessing:
            word_length_limit: 20
    -
        name: body
        type: text
        preprocessing:
            word_length_limit: 2000

```

### Tokenizers

Several different features perform raw data preprocessing by tokenizing strings (for instance sequence, text and set).
Here are the tokenizers options you can specify for those features:

- `characters`: splits every character of the input string in a separate token.
- `space`: splits on space characters using the regex `\s+`.
- `space_punct`: splits on space characters and punctuation using the regex `\w+|[^\w\s]`. 
- `underscore`: splits on the underscore character `_`.
- `comma`: splits on the underscore character `,`.
- `untokenized`: treats the whole string as a single token.
- `stripped`: treats the whole string as a single token after removing spaces at the beginnign and at the end of the string.
- `hf_tokenizer`: uses the Hugging Face AutoTokenizer which uses a `pretrained_model_name_or_path` parameter to decide which tokenizer to load.
- Language specific tokenizers: [spaCy](https://spacy.io) based language tokenizers.

The [spaCy](https://spacy.io) based tokenizers are functions that use the powerful tokenization and NLP preprocessing models provided the library.
Several languages are available: English (code `en`), Italian (code `it`), Spanish (code `es`), German (code `de`), French (code `fr`), Portuguese (code `pt`), Dutch (code `nl`), Greek (code `el`), Chinese (code `zh`), Danish (code `da`), Dutch (code `el`), Japanese (code `ja`), Lithuanian (code `lt`), Norwegian (code `nb`), Polish (code `pl`), Romanian (code `ro`) and Multi (code `xx`, useful in case you have a dataset containing different languages).
For each language different functions are available:

- `tokenize`: uses spaCy tokenizer,
- `tokenize_filter`: uses spaCy tokenizer and filters out punctuation, numbers, stopwords and words shorter than 3 characters,
- `tokenize_remove_stopwords`: uses spaCy tokenizer and filters out stopwords,
- `lemmatize`: uses spaCy lemmatizer,
- `lemmatize_filter`: uses spaCy lemmatizer and filters out punctuation, numbers, stopwords and words shorter than 3 characters,
- `lemmatize_remove_stopwords`: uses spaCy lemmatize and filters out stopwords.

In order to use these options, you have to download the the spaCy model:

```
python -m spacy download <language_code>
```

and provide `<language>_<function>` as `tokenizer` like: `english_tokenizer`, `italian_lemmatize_filter`, `multi_tokenize_filter` and so on.
More details on the models can be found in the [spaCy documentation](https://spacy.io/models).

 
Binary Features
---------------

### Binary Features Preprocessing

Binary features are directly transformed into a binary valued vector of length `n` (where `n` is the size of the dataset) and added to the HDF5 with a key that reflects the name of column in the dataset.
No additional information about them is available in the JSON metadata file.

The parameters available for preprocessing are

- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `0`): the value to replace the missing values with in case the `missing_value_strategy` is `fill_with_const`.

### Binary Input Features and Encoders

Binary features have two encoders.
One encoder (`passthrough'`) takes the raw binary values coming from the input placeholders are just returned as outputs.
Inputs are of size `b` while outputs are of size `b x 1` where `b` is the batch size.
The other encoder (`'dense'`) passes the raw binary values through a fully connected layers.
In this case the inputs of size `b` are transformed to size `b x h`.  

Example binary feature entry in the input features list:

```yaml
name: binary_column_name
type: binary
encoder: passthrough
```

Binary input feature parameters are

- `encoder` (default `'passthrough'`) encodes the binary feature. Valid choices:  `'passthrough'`: binary feature is passed through as-is, `'dense'`: binary feature is fed through a fully connected layer.


There are no additional parameters for the `passthrough` encoder.

### Dense Encoder Parameters

For the `dense` encoder these are the available parameters.

- `num_layers` (default `1`): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): f a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

### Binary Output Features and Decoders

Binary features can be used when a binary classification needs to be performed or when the output is a single probability.
There is only one decoder available for binary features and it is a (potentially empty) stack of fully connected layers, followed by a projection into a single number followed by a sigmoid function.

These are the available parameters of a binary output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to [Output Features Dependencies](#output-features-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `loss` (default `{type: cross_entropy, confidence_penalty: 0, robust_lambda: 0, positive_class_weight: 1}`): is a dictionary containing a loss `type` and its hyper-parameters. The only available loss `type` is `cross_entropy` (cross entropy), and the optional parameters are `confidence_penalty` (an additional term that penalizes too confident predictions by adding a `a * (max_entropy - entropy) / max_entropy` term to the loss, where a is the value of this parameter), `robust_lambda` (replaces the loss with `(1 - robust_lambda) * loss + robust_lambda / 2` which is useful in case of noisy labels) and `positive_class_weight` (multiplies the loss for the positive class, increasing its importance).

These are the available parameters of a binary output feature decoder

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `dropout` (default `0`): dropout rate
- `use_base` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`): initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix. Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `threshold` (default `0.5`): The threshold above (greater or equal) which the predicted output of the sigmoid will be mapped to 1.

Example binary feature entry (with default parameters) in the output features list:

```yaml
name: binary_column_name
type: binary
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    positive_class_weight: 1
fc_layers: null
num_fc_layers: 0
fc_size: 256
activation: relu
norm: null
dropout: 0.2
weisghts_intializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: l1
bias_regularizer: l1
threshold: 0.5
```

### Binary Features Measures

The only measures that are calculated every epoch and are available for binary features are the `accuracy` and the `loss` itself.
You can set either of them as `validation_measure` in the `training` section of the configuration if you set the `validation_field` to be the name of a binary feature.

Numerical Features
------------------

### Numerical Features Preprocessing

Numerical features are directly transformed into a float valued vector of length `n` (where `n` is the size of the dataset) and added to the HDF5 with a key that reflects the name of column in the dataset.
No additional information about them is available in the JSON metadata file.

Parameters available for preprocessing are

- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `0`): the value to replace the missing values with in case the `missing_value_strategy` is `fill-value`.
- `normalization` (default `null`): technique to be used when normalizing the numerical feature types. The available options are `null`, `zscore`, `minmax` and `log1p`. If the value is `null` no normalization is performed. If the value is `zscore`, the mean and standard deviation are computed so that values are shifted to have zero mean and 1 standard deviation. If the value is `minmax`, minimun and maximum values are computed and the minimum is subtracted from values and the result is divided by difference between maximum and minimum.  If `normalization` is `log1p` the  value returned is the natural log of 1 plus the original value.  Note: `log1p` is defined only for positive values.

### Numerical Input Features and Encoders

Numerical features have two encoders.
One encoder (`passthrough'`) takes the raw binary values coming from the input placeholders are just returned as outputs.
Inputs are of size `b` while outputs are of size `b x 1` where `b` is the batch size.
The other encoder (`'dense'`) passes the raw binary values through fully connected layers.
In this case the inputs of size `b` are transformed to size `b x h`.  

The available encoder parameters are:

- `norm'` (default `null`): norm to apply after the single neuron. It can be `null`, `batch` or `layer`.
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of a feature of the same type and with the same encoder parameters.

There are no additional parameters for the `passthrough` encoder.

### Dense Encoder Parameters

For the `dense` encoder these are the available parameters.

- `num_layers` (default `1`): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): f a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate


Example numerical feature entry in the input features list:

```yaml
name: numerical_column_name
type: numerical
norm: null
tied_weights: null
encoder: dense
num_layers: 1
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activation: relu
dropout: 0
```

### Numerical Output Features and Decoders

Numerical features can be used when a regression needs to be performed.
There is only one decoder available for numerical features and it is a (potentially empty) stack of fully connected layers, followed by a projection into a single number.

These are the available parameters of a numerical output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to [Output Features Dependencies](#output-features-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `loss` (default `{type: mean_squared_error}`): is a dictionary containing a loss `type`. The available losses `type` are `mean_squared_error` and `mean_absolute_error`.

These are the available parameters of a numerical output feature decoder

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `dropout` (default `0`): dropout rate
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `clip` (default `null`): If not `null` it specifies a minimum and maximum value the predictions will be clipped to. The value can be either a list or a tuple of length 2, with the first value representing the minimum and the second the maximum. For instance `(-5,5)` will make it so that all predictions will be clipped in the `[-5,5]` interval.

Example numerical feature entry (with default parameters) in the output features list:

```yaml
name: numerical_column_name
type: numerical
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: mean_squared_error
fc_layers: null
num_fc_layers: 0
fc_size: 256
activation: relu
norm: null
norm_params: null
dropout: 0
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
clip: null
```

### Numerical Features Measures

The measures that are calculated every epoch and are available for numerical features are `mean_squared_error`, `mean_absolute_error`, `r2` and the `loss` itself.
You can set either of them as `validation_measure` in the `training` section of the configuration if you set the `validation_field` to be the name of a numerical feature.

Category Features
-----------------

### Category Features Preprocessing

Category features are transformed into an integer valued vector of size `n` (where `n` is the size of the dataset) and added to the HDF5 with a key that reflects the name of column in the dataset.
The way categories are mapped into integers consists in first collecting a dictionary of all the different category strings present in the column of the dataset, then ranking them by frequency and then assigning them an increasing integer ID from the most frequent to the most rare (with 0 being assigned to a `<UNK>` token).
The column name is added to the JSON file, with an associated dictionary containing

1. the mapping from integer to string (`idx2str`)
2. the mapping from string to id (`str2idx`)
3. the mapping from string to frequency (`str2freq`)
4. the size of the set of all tokens (`vocab_size`)
5. additional preprocessing information (by default how to fill missing values and what token to use to fill missing values)

The parameters available for preprocessing are

- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `"<UNK>"`): the value to replace the missing values with in case the `missing_value_strategy` is `fill-value`.
- `lowercase` (default `false`): if the string has to be lowercased before being handled by the tokenizer.
- `most_common` (default `10000`): the maximum number of most common tokens to be considered. if the data contains more than this amount, the most infrequent tokens will be treated as unknown.

### Category Input Features and Encoders

Category features have three encoders.
The `passthrough` encoder passes the raw integer values coming from the input placeholders to outputs of size `b x 1`.
The other two encoders map to either `dense` or `sparse` embeddings (one-hot encodings) and returned as outputs of size `b x h`, where `b` is the batch size and `h` is the dimenionsality of the embeddings.

Input feature parameters.

- `encoder'` (default `dense`): the possible values are `passthrough`, `dense` and `sparse`. `passthrough` means passing the raw integer values unaltered.  `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of a feature of the same type and with the same encoder parameters.

Example binary feature entry in the input features list:

```yaml
name: category_column_name
type: category
tied_weights: null
encoder: dense
```

The available encoder parameters:

#### Dense Encoder

- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embedding plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false` embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `dropout` (default `0`): dropout rate.
- `embedding_initializer` (default `null`): the initializer to use. If `null`, the default initialized of each variable is used (`glorot_uniform` in most cases). Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `embedding_regularizer` (default `null`): specifies the type of regularizer to use `l1`, `l2` or `l1_l2`.


#### Sparse Encoder

- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embedding plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false` embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `dropout` (default `false`): determines if there should be a dropout layer after embedding.
- `initializer` (default `null`): the initializer to use. If `null`, the default initialized of each variable is used (`glorot_uniform` in most cases). Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `regularize` (default `true`): if `true` the embedding weights are added to the set of weights that get regularized by a regularization loss (if the `regularization_lambda` in `training` is greater than 0).
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of a feature of the same type and with the same encoder parameters.

Example category feature entry in the input features list:

```yaml
name: category_column_name
type: category
encoder: sparse
tied_weights: null
embedding_size: 256
embeddings_on_cpu: false
pretrained_embeddings: null
embeddings_trainable: true
dropout: 0
initializer: null
regularizer: null
```

### Category Output Features and Decoders

Category features can be used when a multi-class classification needs to be performed.
There is only one decoder available for category features and it is a (potentially empty) stack of fully connected layers, followed by a projection into a vector of size of the number of available classes, followed by a softmax.

```
+--------------+   +---------+   +-----------+
|Combiner      |   |Fully    |   |Projection |   +-------+
|Output        +--->Connected+--->into Output+--->Softmax|
|Representation|   |Layers   |   |Space      |   +-------+
+--------------+   +---------+   +-----------+
```

These are the available parameters of a category output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to [Output Features Dependencies](#output-features-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `loss` (default `{type: softmax_cross_entropy, class_similarities_temperature: 0, class_weights: 1, confidence_penalty: 0, distortion: 1, labels_smoothing: 0, negative_samples: 0, robust_lambda: 0, sampler: null, unique: false}`): is a dictionary containing a loss `type`. The available losses `type` are `softmax_cross_entropy` and `sampled_softmax_cross_entropy`.
- `top_k` (default `3`): determines the parameter `k`, the number of categories to consider when computing the `top_k` measure. It computes accuracy but considering as a match if the true category appears in the first `k` predicted categories ranked by decoder's confidence.

These are the `loss` parameters

- `confidence_penalty` (default `0`): penalizes overconfident predictions (low entropy) by adding an additional term that penalizes too confident predictions by adding a `a * (max_entropy - entropy) / max_entropy` term to the loss, where a is the value of this parameter. Useful in case of noisy labels.
- `robust_lambda` (default `0`): replaces the loss with `(1 - robust_lambda) * loss + robust_lambda / c` where `c` is the number of classes, which is useful in case of noisy labels.
- `class_weights` (default `1`): the value can be a vector of weights, one for each class, that is multiplied to the loss of the datapoints that have that class as ground truth. It is an alternative to oversampling in case of unbalanced class distribution. The ordering of the vector follows the category to integer ID mapping in the JSON metadata file (the `<UNK>` class needs to be included too). Alternatively, the value can be a dictionary with class strings as keys and weights as values, like `{class_a: 0.5, class_b: 0.7, ...}`.
- `class_similarities` (default `null`): if not `null` it is a `c x c` matrix in the form of a list of lists that contains the mutual similarity of classes. It is used if `class_similarities_temperature` is greater than 0. The ordering of the vector follows the category to integer ID mapping in the JSON metadata file (the `<UNK>` class needs to be included too).
- `class_similarities_temperature` (default `0`): is the temperature parameter of the softmax that is performed on each row of `class_similarities`. The output of that softmax is used to determine the supervision vector to provide instead of the one hot vector that would be provided otherwise for each datapoint. The intuition behind it is that errors between similar classes are more tollerable than errors between really different classes.
- `labels_smoothing` (default `0`): If label_smoothing is nonzero, smooth the labels towards `1/num_classes`: `new_onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes`.
- `negative_samples` (default `0`): if `type` is `sampled_softmax_cross_entropy`, this parameter indicates how many negative samples to use.
- `sampler` (default `null`): options are `fixed_unigram`, `uniform`, `log_uniform`, `learned_unigram`. For a detailed description of the samplers refer to [TensorFlow's documentation](https://www.tensorflow.org/api_guides/python/nn#Candidate_Sampling).
- `distortion` (default `1`): when `loss` is `sampled_softmax_cross_entropy` and the sampler is either `unigram` or `learned_unigram` this is used to skew the unigram probability distribution. Each weight is first raised to the distortion's power before adding to the internal unigram distribution. As a result, distortion = 1.0 gives regular unigram sampling (as defined by the vocab file), and distortion = 0.0 gives a uniform distribution.
- `unique` (default `false`): Determines whether all sampled classes in a batch are unique.

These are the available parameters of a category output feature decoder

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `weights_initializer` and `weighs_regularizer`. If any of those values is missing from the dictionary, the default value will be used.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `dropout` (default `false`): determines if there should be a dropout layer after each layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the fully connected weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the fully connected weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.

Example category feature entry (with default parameters) in the output features list:

```yaml
name: category_column_name
type: category
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: 1
    class_similarities: null
    class_similarities_temperature: 0
    labels_smoothing: 0
    negative_samples: 0
    sampler: null
    distortion: 1
    unique: false
fc_layers: null
num_fc_layers: 0
fc_size: 256
activation: relu
norm: null
norm_params: null
dropout: 0
use_biase: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
top_k: 3
```

### Category Features Measures

The measures that are calculated every epoch and are available for category features are `accuracy`, `top_k` (computes accuracy considering as a match if the true category appears in the first `k` predicted categories ranked by decoder's confidence) and the `loss` itself.
You can set either of them as `validation_measure` in the `training` section of the configuration if you set the `validation_field` to be the name of a category feature.

Set Features
------------

### Set Features Preprocessing

Set features are expected to be provided as a string of elements separated by whitespace, e.g. "elem5 elem9 elem6".
The string values are transformed into a binary (int8 actually) valued matrix of size `n x l` (where `n` is the size of the dataset and `l` is the minimum of the size of the biggest set and a `max_size` parameter) and added to HDF5 with a key that reflects the name of column in the dataset.
The way sets are mapped into integers consists in first using a tokenizer to map from strings to sequences of set items (by default this is done by splitting on spaces).
Then a dictionary of all the different set item strings present in the column of the dataset is collected, then they are ranked by frequency and an increasing integer ID is assigned to them from the most frequent to the most rare (with 0 being assigned to `<PAD>` used for padding and 1 assigned to `<UNK>` item).
The column name is added to the JSON file, with an associated dictionary containing

1. the mapping from integer to string (`idx2str`)
2. the mapping from string to id (`str2idx`)
3. the mapping from string to frequency (`str2freq`)
4. the maximum size of all sets (`max_set_size`)
5. additional preprocessing information (by default how to fill missing values and what token to use to fill missing values)

The parameters available for preprocessing are

- `tokenizer` (default `space`): defines how to map from the raw string content of the dataset column to a set of elements. The default value `space` splits the string on spaces. Common options include: `underscore` (splits on underscore), `comma`(splits on comma), `json` (decodes the string into a set or a list through a JSON parser). For all the available options refer to the [Tokenizers](#tokenizers) section.
- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `0`): the value to replace the missing values with in case the `missing_value_strategy` is `fill-value`.
- `lowercase` (default `false`): if the string has to be lowercased before being handled by the tokenizer.
- `most_common` (default `10000`): the maximum number of most common tokens to be considered. if the data contains more than this amount, the most infrequent tokens will be treated as unknown.

### Set Input Features and Encoders

Set features have one encoder, the raw binary values coming from the input placeholders are first transformed in sparse integer lists, then they are mapped to either dense or sparse embeddings (one-hot encodings), finally they are aggregated and returned as outputs.
Inputs are of size `b` while outputs are of size `b x h` where `b` is the batch size and `h` is the dimensionally of the embeddings.

```
+-+
|0|          +-----+
|0|   +-+    |emb 2|   +-----------+
|1|   |2|    +-----+   |Aggregation|
|0+--->4+---->emb 4+--->Reduce     +->
|1|   |5|    +-----+   |Operation  |
|1|   +-+    |emb 5|   +-----------+
|0|          +-----+
+-+
```

The available encoder parameters are

- `representation'` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `50`): it is the maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false` embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embedding plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `weights_initializer` and `weighs_regularizer`. If any of those values is missing from the dictionary, the default value will be used.
- `num_fc_layers` (default `1`): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `10`): f a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `reduce_output` (default `sum`): describes the strategy to use to aggregate the embeddings of the items of the set. Possible values are `sum`, `mean` and `sqrt` (the weighted sum divided by the square root of the sum of the squares of the weights).
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of a feature of the same type and with the same encoder parameters.

Example set feature entry in the input features list:

```yaml
name: set_column_name
type: set
representation: dense
embedding_size: 50
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0.0
reduce_output: sum
tied_weights: null
```

### Set Output Features and Decoders

Set features can be used when multi-label classification needs to be performed.
There is only one decoder available for set features and it is a (potentially empty) stack of fully connected layers, followed by a projection into a vector of size of the number of available classes, followed by a sigmoid.

```
+--------------+   +---------+   +-----------+
|Combiner      |   |Fully    |   |Projection |   +-------+
|Output        +--->Connected+--->into Output+--->Sigmoid|
|Representation|   |Layers   |   |Space      |   +-------+
+--------------+   +---------+   +-----------+
```

These are the available parameters of the set output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to [Output Features Dependencies](#output-features-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `loss` (default `{type: sigmoid_cross_entropy}`): is a dictionary containing a loss `type`. The available loss `type` is `sigmoid_cross_entropy`.

These are the available parameters of a set output feature decoder

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): f a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `threshold` (default `0.5`): The threshold above (greater or equal) which the predicted output of the sigmoid will be mapped to 1.

Example set feature entry (with default parameters) in the output features list:

```yaml
name: set_column_name
type: set
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: sigmoid_cross_entropy
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0.0
threshold: 0.5
```

### Set Features Measures

The measures that are calculated every epoch and are available for category features are `jaccard_index` and the `loss` itself.
You can set either of them as `validation_measure` in the `training` section of the configuration if you set the `validation_field` to be the name of a set feature.

Bag Features
------------

### Bag Features Preprocessing

Bag features are expected to be provided as a string of elements separated by whitespace, e.g. "elem5 elem9 elem6".
Bag features are treated in the same way of set features, with the only difference being that the matrix had float values (frequencies).

### Bag Input Features and Encoders

Bag features have one encoder, the raw float values coming from the input placeholders are first transformed in sparse integer lists, then they are mapped to either dense or sparse embeddings (one-hot encodings), they are aggregated as a weighted sum, where the weights are the original float values, and finally returned as outputs.
Inputs are of size `b` while outputs are of size `b x h` where `b` is the batch size and `h` is the dimensionality of the embeddings.

The parameters are the same used for set input features with the exception of `reduce_output` that does not apply in this case because the weighted sum already acts as a reducer.

### Bag Output Features and Decoders

There is no bag decoder available yet.

### Bag Features Measures

As there is no decoder there is also no measure available yet for bag feature.

Sequence Features
-----------------

### Sequence Features Preprocessing

Sequence features are transformed into an integer valued matrix of size `n x l` (where `n` is the size of the dataset and `l` is the minimum of the length of the longest sequence and a `sequence_length_limit` parameter) and added to HDF5 with a key that reflects the name of column in the dataset.
The way sequences are mapped into integers consists in first using a tokenizer to map from strings to sequences of tokens (by default this is done by splitting on spaces).
Then a dictionary of all the different token strings present in the column of the dataset is collected, then they are ranked by frequency and an increasing integer ID is assigned to them from the most frequent to the most rare (with 0 being assigned to `<PAD>` used for padding and 1 assigned to `<UNK>` item).
The column name is added to the JSON file, with an associated dictionary containing

1. the mapping from integer to string (`idx2str`)
2. the mapping from string to id (`str2idx`)
3. the mapping from string to frequency (`str2freq`)
4. the maximum length of all sequences (`sequence_length_limit`)
5. additional preprocessing information (by default how to fill missing values and what token to use to fill missing values)

The parameters available for preprocessing are

- `sequence_length_limit` (default `256`): the maximum length of the sequence. Sequences that are longer than this value will be truncated, while sequences that are shorter will be padded.
- `most_common` (default `20000`): the maximum number of most common tokens to be considered. if the data contains more than this amount, the most infrequent tokens will be treated as unknown.
- `padding_symbol` (default `<PAD>`): the string used as a padding symbol. Is is mapped to the integer ID 0 in the vocabulary.
- `unknown_symbol` (default `<UNK>`): the string used as a unknown symbol. Is is mapped to the integer ID 1 in the vocabulary.
- `padding` (default `right`): the direction of the padding. `right` and `left` are available options.
- `tokenizer` (default `space`): defines how to map from the raw string content of the dataset column to a sequence of elements. For the available options refer to the [Tokenizers](#tokenizers)section.
- `lowercase` (default `false`): if the string has to be lowercase before being handled by the tokenizer.
- `vocab_file` (default `null`)  filepath string to a UTF-8 encoded file containing the sequence's vocabulary.  On each line the first string until `\t` or `\n` is considered a word.
- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `""`): the value to replace the missing values with in case the `missing_value_strategy` is `fill_value`.

### Sequence Input Features and Encoders

Sequence features have several encoders and each of them has its own parameters.
Inputs are of size `b` while outputs are of size `b x h` where `b` is the batch size and `h` is the dimensionally of the output of the encoder.
In case a representation for each element of the sequence is needed (for example for tagging them, or for using an attention mechanism), one can specify the parameter `reduce_output` to be  `null` and the output will be a `b x s x h` tensor where `s` is the length of the sequence.
Some encoders, because of their inner workings, may require additional parameters to be specified in order to obtain one representation for each element of the sequence.
For instance the `parallel_cnn` encoder, by default pools and flattens the sequence dimension and then passes the flattened vector through fully connected layers, so in order to obtain the full tesnor one has to specify `reduce_output: null`.

Sequence input feature parameters are

- `encoder` (default ``parallel_cnn``): the name of the encoder to use to encode the sequence. The available ones are  `embed`, `parallel_cnn`, `stacked_cnn`, `stacked_parallel_cnn`, `rnn`, `cnnrnn`, `transformer` and `passthrough` (equivalent to specify `null` or `'None'`).
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of a feature of the same type and with the same encoder parameters.

#### Embed Encoder

The embed encoder simply maps each integer in the sequence to an embedding, creating a `b x s x h` tensor where `b` is the batch size, `s` is the length of the sequence and `h` is the embedding size.
The tensor is reduced along the `s` dimension to obtain a single vector of size `h` for each element of the batch.
If you want to output the full `b x s x h` tensor, you can specify `reduce_output: null`.

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|   +-----------+
|43|   +------+   |Aggregation|
|65+--->Emb 65+--->Reduce     +->
|23|   +------+   |Operation  |
|4 |   |Emb 23|   +-----------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+
```

These are the parameters available for the embed encoder

- `representation'` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false` embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embedding plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `dropout` (default `0`): dropout rate.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `reduce_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).

Example sequence feature entry in the input features list using an embed encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: embed
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
dropout: 0
weights_initializer: null
weights_regularizer: null
reduce_output: sum
```

#### Parallel CNN Encoder

The parallel cnn encoder is inspired by [Yoon Kim's Convolutional Neural Network for Sentence Classification](https://arxiv.org/abs/1408.5882).
It works by first mapping the input integer sequence `b x s` (where `b` is the batch size and `s` is the length of the sequence) into a sequence of embeddings, then it passes the embedding through a number of parallel 1d convolutional layers with different filter size (by default 4 layers with filter size 2, 3, 4 and 5), followed by max pooling and concatenation.
This single vector concatenating the outputs of the parallel convolutional layers is then passed through a stack of fully connected layers and returned as a `b x h` tensor where `h` is the output size of the last fully connected layer.
If you want to output the full `b x s x h` tensor, you can specify `reduce_output: null`.

```
                   +-------+   +----+
                +-->1D Conv+--->Pool+-+
       +------+ |  |Width 2|   +----+ |
       |Emb 12| |  +-------+          |
       +------+ |                     |
+--+   |Emb 7 | |  +-------+   +----+ |
|12|   +------+ +-->1D Conv+--->Pool+-+
|7 |   |Emb 43| |  |Width 3|   +----+ |           +---------+
|43|   +------+ |  +-------+          | +------+  |Fully    |
|65+--->Emb 65+-+                     +->Concat+-->Connected+->
|23|   +------+ |  +-------+   +----+ | +------+  |Layers   |
|4 |   |Emb 23| +-->1D Conv+--->Pool+-+           +---------+
|1 |   +------+ |  |Width 4|   +----+ |
+--+   |Emb 4 | |  +-------+          |
       +------+ |                     |
       |Emb 1 | |  +-------+   +----+ |
       +------+ +-->1D Conv+--->Pool+-+
                   |Width 5|   +----+
                   +-------+
```

These are the available for an parallel cnn encoder:

- `representation'` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false` embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embedding plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `conv_layers` (default `null`): it is a list of dictionaries containing the parameters of all the convolutional layers. The length of the list determines the number of parallel convolutional layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `filter_size`, `num_filters`, `pool`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `conv_layers` and `num_conv_layers` are `null`, a default list will be assigned to `conv_layers` with the value `[{filter_size: 2}, {filter_size: 3}, {filter_size: 4}, {filter_size: 5}]`.
- `num_conv_layers` (default `null`): if `conv_layers` is `null`, this is the number of parallel convolutional layers.
- `filter_size` (default `3`): if a `filter_size` is not already specified in `conv_layers` this is the default `filter_size` that will be used for each layer. It indicates how wide is the 1d convolutional filter.
- `num_filters` (default `256`): if a `num_filters` is not already specified in `conv_layers` this is the default `num_filters` that will be used for each layer. It indicates the number of filters, and by consequence the output channels of the 1d convolution.
- `pool_function` (default `max`):  pooling function: `max` will select the maximum value.  Any of these--`average`, `avg` or `mean`--will compute the mean value.
- `pool_size` (default `null`): if a `pool_size` is not already specified in `conv_layers` this is the default `pool_size` that will be used for each layer. It indicates the size of the max pooling that will be performed along the `s` sequence dimension after the convolution operation.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`,  `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `null`): if `fc_layers` is `null`, this is the number of stacked fully connected layers (only applies if `reduce_output` is not `null`).
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `reduce_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the sequence dimension), `last` (returns the last vector of the sequence dimension) and  `null` (which does not reduce and returns the full tensor).

Example sequence feature entry in the input features list using a parallel cnn encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: parallel_cnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_on_cpu: false
pretrained_embeddings: null
embeddings_trainable: true
conv_layers: null
num_conv_layers: null
filter_size: 3
num_filters: 256
pool_function: max
pool_size: null
fc_layers: null
num_fc_layers: null
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0.0
reduce_output: sum
```

#### Stacked CNN Encoder

The stacked cnn encoder is inspired by [Xiang Zhang at all's Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626).
It works by first mapping the input integer sequence `b x s` (where `b` is the batch size and `s` is the length of the sequence) into a sequence of embeddings, then it passes the embedding through a stack of 1d convolutional layers with different filter size (by default 6 layers with filter size 7, 7, 3, 3, 3 and 3), followed by an optional final pool and by a flatten operation.
This single flatten vector is then passed through a stack of fully connected layers and returned as a `b x h` tensor where `h` is the output size of the last fully connected layer.
If you want to output the full `b x s x h` tensor, you can specify the `pool_size` of all your `conv_layers` to be `null`  and `reduce_output: null`, while if `pool_size` has a value different from `null` and `reduce_output: null` the returned tensor will be of shape `b x s' x h`, where `s'` is width of the output of the last convolutional layer.

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|   +----------------+  +---------+
|43|   +------+   |1D Conv         |  |Fully    |
|65+--->Emb 65+--->Layers          +-->Connected+->
|23|   +------+   |Different Widths|  |Layers   |
|4 |   |Emb 23|   +----------------+  +---------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+
```

These are the parameters available for the stack cnn encoder:

- `representation'` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false` embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embedding plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `conv_layers` (default `null`): it is a list of dictionaries containing the parameters of all the convolutional layers. The length of the list determines the number of stacked convolutional layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `filter_size`, `num_filters`, `pool_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `conv_layers` and `num_conv_layers` are `null`, a default list will be assigned to `conv_layers` with the value `[{filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: true}, {filter_size: 3, pool_size: 3, regularize: true}]`.
- `num_conv_layers` (default `null`): if `conv_layers` is `null`, this is the number of stacked convolutional layers.
- `filter_size` (default `3`): if a `filter_size` is not already specified in `conv_layers` this is the default `filter_size` that will be used for each layer. It indicates how wide is the 1d convolutional filter.
- `num_filters` (default `256`): if a `num_filters` is not already specified in `conv_layers` this is the default `num_filters` that will be used for each layer. It indicates the number of filters, and by consequence the output channels of the 1d convolution.
- `strides` (default `1`): stride length of the convolution
- `padding` (default `same`):  one of `valid` or `same`.
- `dilation_rate` (default `1`): dilation rate to use for dilated convolution
- `pool_function` (default `max`):  pooling function: `max` will select the maximum value.  Any of these--`average`, `avg` or `mean`--will compute the mean value.
- `pool_size` (default `null`): if a `pool_size` is not already specified in `conv_layers` this is the default `pool_size` that will be used for each layer. It indicates the size of the max pooling that will be performed along the `s` sequence dimension after the convolution operation.
- `pool_strides` (default `null`): factor to scale down
- `pool_padding` (default `same`): one of `valid` or `same`
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `null`): if `fc_layers` is `null`, this is the number of stacked fully connected layers (only applies if `reduce_output` is not `null`).
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `reduce_output` (default `max`): defines how to reduce the output tensor of the convolutional layers along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).

Example sequence feature entry in the input features list using a parallel cnn encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: stacked_cnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
conv_layers: null
num_conv_layers: null
filter_size: 3
num_filters: 256
strides: 1
padding: same
dilation_rate: 1
pool_function: max
pool_size: null
pool_strides: null
pool_padding: same
fc_layers: null
num_fc_layers: null
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
reduce_output: max
```

#### Stacked Parallel CNN Encoder

The stacked parallel cnn encoder is a combination of the Parallel CNN and the Stacked CNN encoders where each layer of the stack is a composed of parallel convolutional layers.
It works by first mapping the input integer sequence `b x s` (where `b` is the batch size and `s` is the length of the sequence) into a sequence of embeddings, then it passes the embedding through a stack of several parallel 1d convolutional layers with different filter size, followed by an optional final pool and by a flatten operation.
This single flatten vector is then passed through a stack of fully connected layers and returned as a `b x h` tensor where `h` is the output size of the last fully connected layer.
If you want to output the full `b x s x h` tensor, you can specify `reduce_output: null`.

```
                   +-------+                      +-------+
                +-->1D Conv+-+                 +-->1D Conv+-+
       +------+ |  |Width 2| |                 |  |Width 2| |
       |Emb 12| |  +-------+ |                 |  +-------+ |
       +------+ |            |                 |            |
+--+   |Emb 7 | |  +-------+ |                 |  +-------+ |
|12|   +------+ +-->1D Conv+-+                 +-->1D Conv+-+
|7 |   |Emb 43| |  |Width 3| |                 |  |Width 3| |                   +---------+
|43|   +------+ |  +-------+ | +------+  +---+ |  +-------+ | +------+  +----+  |Fully    |
|65+--->Emb 65+-+            +->Concat+-->...+-+            +->Concat+-->Pool+-->Connected+->
|23|   +------+ |  +-------+ | +------+  +---+ |  +-------+ | +------+  +----+  |Layers   |
|4 |   |Emb 23| +-->1D Conv+-+                 +-->1D Conv+-+                   +---------+
|1 |   +------+ |  |Width 4| |                 |  |Width 4| |
+--+   |Emb 4 | |  +-------+ |                 |  +-------+ |
       +------+ |            |                 |            |
       |Emb 1 | |  +-------+ |                 |  +-------+ |
       +------+ +-->1D Conv+-+                 +-->1D Conv+-+
                   |Width 5|                      |Width 5|
                   +-------+                      +-------+
```

These are the available parameters for the stack parallel cnn encoder:

- `representation'` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false` embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embedding plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `stacked_layers` (default `null`): it is a of lists of list of dictionaries containing the parameters of the stack of parallel convolutional layers. The length of the list determines the number of stacked parallel convolutional layers, length of the sub-lists determines the number of parallel conv layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `filter_size`, `num_filters`, `pool_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `stacked_layers` and `num_stacked_layers` are `null`, a default list will be assigned to `stacked_layers` with the value `[[{filter_size: 2}, {filter_size: 3}, {filter_size: 4}, {filter_size: 5}], [{filter_size: 2}, {filter_size: 3}, {filter_size: 4}, {filter_size: 5}], [{filter_size: 2}, {filter_size: 3}, {filter_size: 4}, {filter_size: 5}]]`.
- `num_stacked_layers` (default `null`): if `stacked_layers` is `null`, this is the number of elements in the stack of parallel convolutional layers.
- `filter_size` (default `3`): if a `filter_size` is not already specified in `conv_layers` this is the default `filter_size` that will be used for each layer. It indicates how wide is the 1d convolutional filter.
- `num_filters` (default `256`): if a `num_filters` is not already specified in `conv_layers` this is the default `num_filters` that will be used for each layer. It indicates the number of filters, and by consequence the output channels of the 1d convolution.
- `pool_function` (default `max`):  pooling function: `max` will select the maximum value.  Any of these--`average`, `avg` or `mean`--will compute the mean value.
- `pool_size` (default `null`): if a `pool_size` is not already specified in `conv_layers` this is the default `pool_size` that will be used for each layer. It indicates the size of the max pooling that will be performed along the `s` sequence dimension after the convolution operation.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `null`): if `fc_layers` is `null`, this is the number of stacked fully connected layers (only applies if `reduce_output` is not `null`).
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `reduce_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).

Example sequence feature entry in the input features list using a parallel cnn encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: stacked_parallel_cnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
stacked_layers: null
num_stacked_layers: null
filter_size: 3
num_filters: 256
pool_function: max
pool_size: null
fc_layers: null
num_fc_layers: null
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
reduce_output: max
```

#### RNN Encoder

The rnn encoder works by first mapping the input integer sequence `b x s` (where `b` is the batch size and `s` is the length of the sequence) into a sequence of embeddings, then it passes the embedding through a stack of recurrent layers (by default 1 layer), followed by a reduce operation that by default only returns the last output, but can perform other reduce functions.
If you want to output the full `b x s x h` where `h` is the size of the output of the last rnn layer, you can specify `reduce_output: null`.

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|                 +---------+
|43|   +------+   +----------+  |Fully    |
|65+--->Emb 65+--->RNN Layers+-->Connected+->
|23|   +------+   +----------+  |Layers   |
|4 |   |Emb 23|                 +---------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+


```

These are the available parameters for the rnn encoder:

- `representation'` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false` embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embedding plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `num_layers` (default `1`): the number of stacked recurrent layers.
- `state_size` (default `256`): the size of the state of the rnn.
- `cell_type` (default `rnn`): the type of recurrent cell to use. Available values are: `rnn`, `lstm`, `lstm_block`, `lstm`, `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`. For reference about the differences between the cells please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell). We suggest to use the `block` variants on CPU and the `cudnn` variants on GPU because of their increased speed.
- `bidirectional` (default `false`): if `true` two recurrent networks will perform encoding in the forward and backward direction and their outputs will be concatenated.
- `activation` (default `'tanh'`): activation function to use
- `recurrent_activation` (default `'sigmoid'`): activation function to use in the recurrent step
- `unit_forget_bias` (default `true`): If `true`, add 1 to the bias of the forget gate at initialization
- `recurrent_initializer` (default `'orthogonal'`): initializer for recurrent matrix weights
- `recurrent_regularizer` (default `null`): regularizer function applied to recurrent matrix weights
- `dropout` (default `0.0`): dropout rate
- `recurrent_dropout` (default `0.0`): dropout rate for recurrent state
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`,  `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `null`): if `fc_layers` is `null`, this is the number of stacked fully connected layers (only applies if `reduce_output` is not `null`).
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `fc_activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `fc_dropout` (default `0`): dropout rate
- `reduce_output` (default `last`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).

Example sequence feature entry in the input features list using a parallel cnn encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: rnn
tied_weights: null
representation': dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
num_layers: 1
state_size: 256
cell_type: rnn
bidirectional: false
activation: tanh
recurrent_activation: sigmoid
unit_forget_bias: true
recurrent_initializer: orthogonal
recurrent_regularizer: null
dropout: 0.0
recurrent_dropout: 0.0
fc_layers: null
num_fc_layers: null
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
reduce_output: last
```

#### CNN RNN Encoder

The `cnnrnn` encoder works by first mapping the input integer sequence `b x s` (where `b` is the batch size and `s` is the length of the sequence) into a sequence of embeddings, then it passes the embedding through a stack of convolutional layers (by default 2), that is followed by a stack of recurrent layers (by default 1), followed by a reduce operation that by default only returns the last output, but can perform other reduce functions.
If you want to output the full `b x s x h` where `h` is the size of the output of the last rnn layer, you can specify `reduce_output: null`.

```
       +------+
       |Emb 12|
       +------+
+--+   |Emb 7 |
|12|   +------+
|7 |   |Emb 43|                                +---------+
|43|   +------+   +----------+   +----------+  |Fully    |
|65+--->Emb 65+--->CNN Layers+--->RNN Layers+-->Connected+->
|23|   +------+   +----------+   +----------+  |Layers   |
|4 |   |Emb 23|                                +---------+
|1 |   +------+
+--+   |Emb 4 |
       +------+
       |Emb 1 |
       +------+
```

These are the available parameters of the cnn rnn encoder:

- `representation'` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false` embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embedding plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `conv_layers` (default `null`): it is a list of dictionaries containing the parameters of all the convolutional layers. The length of the list determines the number of stacked convolutional layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `filter_size`, `num_filters`, `pool_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `conv_layers` and `num_conv_layers` are `null`, a default list will be assigned to `conv_layers` with the value `[{filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: true}, {filter_size: 3, pool_size: 3, regularize: true}]`.
- `num_conv_layers` (default `1`): the number of stacked convolutional layers.
- `num_filters` (default `256`): if a `num_filters` is not already specified in `conv_layers` this is the default `num_filters` that will be used for each layer. It indicates the number of filters, and by consequence the output channels of the 1d convolution.
- `filter_size` (default `5`): if a `filter_size` is not already specified in `conv_layers` this is the default `filter_size` that will be used for each layer. It indicates how wide is the 1d convolutional filter.
- `strides` (default `1`): stride length of the convolution
- `padding` (default `same`):  one of `valid` or `same`.
- `dilation_rate` (default `1`): dilation rate to use for dilated convolution
- `conv_activation` (default `relu`): activation for the convolution layer
- `conv_dropout` (default `0.0`): dropout rate for the convolution layer
- `pool_function` (default `max`):  pooling function: `max` will select the maximum value.  Any of these--`average`, `avg` or `mean`--will compute the mean value.
- `pool_size` (default 2 ): if a `pool_size` is not already specified in `conv_layers` this is the default `pool_size` that will be used for each layer. It indicates the size of the max pooling that will be performed along the `s` sequence dimension after the convolution operation.
- `pool_strides` (default `null`): factor to scale down
- `pool_padding` (default `same`): one of `valid` or `same`
- `num_rec_layers` (default `1`): the number of recurrent layers
- `state_size` (default `256`): the size of the state of the rnn.
- `cell_type` (default `rnn`): the type of recurrent cell to use. Available values are: `rnn`, `lstm`, `lstm_block`, `lstm`, `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`. For reference about the differences between the cells please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell). We suggest to use the `block` variants on CPU and the `cudnn` variants on GPU because of their increased speed.
- `bidirectional` (default `false`): if `true` two recurrent networks will perform encoding in the forward and backward direction and their outputs will be concatenated.
- `activation` (default `'tanh'`): activation function to use
- `recurrent_activation` (default `'sigmoid'`): activation function to use in the recurrent step
- `unit_forget_bias` (default `true`): If `true`, add 1 to the bias of the forget gate at initialization
- `recurrent_initializer` (default `'orthogonal'`): initializer for recurrent matrix weights
- `recurrent_regularizer` (default `null`): regularizer function applied to recurrent matrix weights
- `dropout` (default `0.0`): dropout rate
- `recurrent_dropout` (default `0.0`): dropout rate for recurrent state
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`,  `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `null`): if `fc_layers` is `null`, this is the number of stacked fully connected layers (only applies if `reduce_output` is not `null`).
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `fc_activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `fc_dropout` (default `0`): dropout rate
- `reduce_output` (default `last`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).

Example sequence feature entry in the inputs features list using a cnn rnn encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: cnnrnn
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
conv_layers: null
num_conv_layers: 1
num_filters: 256
filter_size: 5
strides: 1
padding: same
dilation_rate: 1
conv_activation: relu
conv_dropout: 0.0
pool_function: max
pool_size: 2
pool_strides: null
pool_padding: same
num_rec_layers: 1
state_size: 256
cell_type: rnn
bidirectional: false
activation: tanh
recurrent_activation: sigmoid
unit_forget_bias: true
recurrent_initializer: orthogonal
recurrent_regularizer: null
dropout: 0.0
recurrent_dropout: 0.0
fc_layers: null
num_fc_layers: null
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
reduce_output: last
```

#### Transformer Encoder

The `transformer` encoder implements a stack of transformer blocks, replicating the architecture introduced in the [Attention is all you need](https://arxiv.org/abs/1706.03762) paper, and adds am optional stack of fully connected layers at the end.

```
       +------+                     
       |Emb 12|                     
       +------+                     
+--+   |Emb 7 |                     
|12|   +------+                     
|7 |   |Emb 43|   +-------------+   +---------+ 
|43|   +------+   |             |   |Fully    |
|65+---+Emb 65+---> Transformer +--->Connected+->
|23|   +------+   | Blocks      |   |Layers   |
|4 |   |Emb 23|   +-------------+   +---------+
|1 |   +------+                     
+--+   |Emb 4 |                     
       +------+                     
       |Emb 1 |                     
       +------+                     

```

- `representation'` (default `dense`): the possible values are `dense` and `sparse`. `dense` means the embeddings are initialized randomly, `sparse` means they are initialized to be one-hot encodings.
- `embedding_size` (default `256`): it is the maximum embedding size, the actual size will be `min(vocabulary_size, embedding_size)` for `dense` representations and exactly `vocabulary_size` for the `sparse` encoding, where `vocabulary_size` is the number of different strings appearing in the training set in the column the feature is named after (plus 1 for `<UNK>`).
- `embeddings_trainable` (default `true`): If `true` embeddings are trained during the training process, if `false` embeddings are fixed. It may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter has effect only when `representation` is `dense` as `sparse` one-hot encodings are not trainable.
- `pretrained_embeddings` (default `null`): by default `dense` embeddings are initialized randomly, but this parameter allows to specify a path to a file containing embeddings in the [GloVe format](https://nlp.stanford.edu/projects/glove/). When the file containing the embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, the others are discarded. If the vocabulary contains strings that have no match in the embeddings file, their embeddings are initialized with the average of all other embedding plus some random noise to make them different from each other. This parameter has effect only if `representation` is `dense`.
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `num_layers` (default `1`): number of transformer blocks.
- `hidden_size` (default `256`): the size of the hidden representation within the transformer block. It is usually the same of the `embedding_size`, but if the two values are different, a projection layer will be added before the first transformer block.
- `num_heads` (default `8`): number of heads of the self attention in the transformer block.
- `transformer_fc_size` (default `256`): Size of the fully connected layer after self attention in the transformer block. This is usually the same as `hidden_size` and `embedding_size`.
- `dropout` (default `0.1`): dropout rate for the transformer block
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`,  `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `0`): This is the number of stacked fully connected layers (only applies if `reduce_output` is not `null`).
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `fc_activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `fc_dropout` (default `0`): dropout rate
- `reduce_output` (default `last`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).

Example sequence feature entry in the inputs features list using a Transformer encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: transformer
tied_weights: null
representation: dense
embedding_size: 256
embeddings_trainable: true
pretrained_embeddings: null
embeddings_on_cpu: false
num_layers: 1
hidden_size: 256
num_heads: 8
transformer_fc_size: 256
dropout: 0.1
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
reduce_output: last
```

| No. | Resources to learn more: |
| ------------- |:-------------:|
| 1.     | [CS480/680 Lecture 19: Attention and Transformer Networks](https://www.youtube.com/watch?v=OyFJWRnt_AY&list=PLk4mwFjvagV3vp1JZ3lNohb1LalMp-VOY&index=1) (VIDEO)     |
| 2.     | [Attention is all you need - Attentional Neural Network Models Masterclass](https://www.youtube.com/watch?v=rBCqOTEfxvg&list=PLk4mwFjvagV3vp1JZ3lNohb1LalMp-VOY&index=2) (VIDEO)    |
| 3.      |  [Illustrated: Self-Attention](https://colab.research.google.com/drive/1rPk3ohrmVclqhH7uQ7qys4oznDdAhpzF) (Colab notebook)     |

#### Passthrough Encoder

The passthrough decoder simply transforms each input value into a float value and adds a dimension to the input tensor, creating a `b x s x 1` tensor where `b` is the batch size and `s` is the length of the sequence.
The tensor is reduced along the `s` dimension to obtain a single vector of size `h` for each element of the batch.
If you want to output the full `b x s x h` tensor, you can specify `reduce_output: null`.
This encoder is not really useful for `sequence` or `text` features, but may be useful for `timeseries` features, as it allows for using them without any processing in later stages of the model, like in a sequence combiner for instance.

```  
+--+   
|12|   
|7 |                    +-----------+
|43|   +------------+   |Aggregation|
|65+--->Cast float32+--->Reduce     +->
|23|   +------------+   |Operation  |
|4 |                    +-----------+
|1 |   
+--+   
```

These are the parameters available for the passthrough encoder

- `reduce_output` (default `null`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).

Example sequence feature entry in the input features list using a passthrough encoder:

```yaml
name: sequence_column_name
type: sequence
encoder: passthrough
reduce_output: null
```

### Sequence Output Features and Decoders

Sequential features can be used when sequence tagging (classifying each element of an input sequence) or sequence generation needs to be performed.
There are two decoders available for those to tasks names `tagger` and `generator`.

These are the available parameters of a sequence output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to [Output Features Dependencies](#output-features-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `loss` (default `{type: softmax_cross_entropy, class_similarities_temperature: 0, class_weights: 1, confidence_penalty: 0, distortion: 1, labels_smoothing: 0, negative_samples: 0, robust_lambda: 0, sampler: null, unique: false}`): is a dictionary containing a loss `type`. The available losses `type` are `softmax_cross_entropy` and `sampled_softmax_cross_entropy`. For details on both losses, please refer to the [category feature output feature section](#category-output-features-and-encoders).

#### Tagger Decoder

In the case of `tagger` the decoder is a (potentially empty) stack of fully connected layers, followed by a projection into a tensor of size `b x s x c`, where `b` is the batch size, `s` is the length of the sequence and `c` is the number of classes, followed by a softmax_cross_entropy.
This decoder requires its input to be shaped as `b x s x h`, where `h` is an hidden dimension, which is the output of a sequence, text or timeseries input feature without reduced outputs or the output of a sequence-based combiner.
If a `b x h` input is provided instead, an error will be raised during model building.

```
Combiner
Output

+---+                 +----------+   +-------+
|emb|   +---------+   |Projection|   |Softmax|
+---+   |Fully    |   +----------+   +-------+
|...+--->Connected+--->...       +--->...    |
+---+   |Layers   |   +----------+   +-------+
|emb|   +---------+   |Projection|   |Softmax|
+---+                 +----------+   +-------+
```

These are the available parameters of a tagger decoder:

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `attention` (default `false`): If `true`, applies a multi-head self attention layer befre prediction.
- `attention_embedding_size` (default `256`): the embedding size of the multi-head self attention layer.
- `attention_num_heads` (default `8`): number of attention heads in the multi-head self attention layer.

Example sequence feature entry using a tagger decoder (with default parameters) in the output features list:

```yaml
name: sequence_column_name
type: sequence
decoder: tagger
reduce_input: null
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: 1
    class_similarities: null
    class_similarities_temperature: 0
    labels_smoothing: 0
    negative_samples: 0
    sampler: null
    distortion: 1
    unique: false
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
attention: false
attention_embedding_size: 256
attention_num_heads: 8
```

#### Generator Decoder

In the case of `generator` the decoder is a (potentially empty) stack of fully connected layers, followed by an rnn that generates outputs feeding on its own previous predictions and generates a tensor of size `b x s' x c`, where `b` is the batch size, `s'` is the length of the generated sequence and `c` is the number of classes, followed by a softmax_cross_entropy.
During training teacher forcing is adopted, meaning the list of targets is provided as both inputs and outputs (shifted by 1), while at evaluation time greedy decoding (generating one token at a time and feeding it as input for the next step) is performed by beam search, using a beam of 1 by default.
By default a generator expects a `b x h` shaped input tensor, where `h` is a hidden dimension.
The `h` vectors are (after an optional stack of fully connected layers) fed into the rnn generator.
One exception is when the generator uses attention, as in that case the expected size of the input tensor is `b x s x h`, which is the output of a sequence, text or timeseries input feature without reduced outputs or the output of a sequence-based combiner.
If a `b x h` input is provided to a generator decoder using an rnn with attention instead, an error will be raised during model building.

```
                            Output     Output
                               1  +-+    ... +--+    END
                               ^    |     ^     |     ^
+--------+   +---------+       |    |     |     |     |
|Combiner|   |Fully    |   +---+--+ | +---+---+ | +---+--+
|Output  +--->Connected+---+RNN   +--->RNN... +--->RNN   |
|        |   |Layers   |   +---^--+ | +---^---+ | +---^--+
+--------+   +---------+       |    |     |     |     |
                              GO    +-----+     +-----+
```


- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).


These are the available parameters of a Generator decoder:

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `cell_type` (default `rnn`): the type of recurrent cell to use. Available values are: `rnn`, `lstm`, `lstm_block`, `lstm`, `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`. For reference about the differences between the cells please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell). We suggest to use the `block` variants on CPU and the `cudnn` variants on GPU because of their increased speed.
- `state_size` (default `256`): the size of the state of the rnn.
- `embedding_size` (default `256`): if `tied_target_embeddings` is `false`, the input embeddings and the weights of the softmax_cross_entropy weights before the softmax_cross_entropy are not tied together and can have different sizes, this parameter describes the size of the embeddings of the inputs of the generator.
- `beam_width` (default `1`): sampling from the rnn generator is performed using beam search. By default, with a beam of one, only a greedy sequence using always the most probably next token is generated, but the beam size can be increased. This usually leads to better performance at the expense of more computation and slower generation.
- `attention` (default `null`): the recurrent generator may use an attention mechanism. The available ones are `bahdanau` and `luong` (for more information refer to [TensorFlow's documentation](https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention)). When `attention` is not `null` the expected size of the input tensor is `b x s x h`, which is the output of a sequence, text or timeseries input feature without reduced outputs or the output of a sequence-based combiner. If a `b x h` input is provided to a generator decoder using an rnn with attention instead, an error will be raised during model building.
- `tied_embeddings` (default `null`): if `null` the embeddings of the targets are initialized randomly, while if the values is the name of an input feature, the embeddings of that input feature will be used as embeddings of the target. The `vocabulary_size` of that input feature has to be the same of the output feature one and it has to have an embedding matrix (binary and numerical features will not have one, for instance). In this case the `embedding_size` will be the same as the `state_size`. This is useful for implementing autoencoders where the encoding and decoding part of the model share parameters.
- `max_sequence_length` (default `0`):

Example sequence feature entry using a generator decoder (with default parameters) in the output features list:

```yaml
name: sequence_column_name
type: sequence
decoder: generator
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: 1
    class_similarities: null
    class_similarities_temperature: 0
    labels_smoothing: 0
    negative_samples: 0
    sampler: null
    distortion: 1
    unique: false
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
cell_type: rnn
state_size: 256
embedding_size: 256
beam_width: 1
attention: null
tied_embeddings: null
max_sequence_length: 0
```

### Sequence Features Measures

The measures that are calculated every epoch and are available for category features are `accuracy` (counts the number of datapoints where all the elements of the predicted sequence are correct over the number of all datapoints), `token_accuracy` (computes the number of elements in all the sequences that are correctly predicted over the number of all the elements in all the sequences), `last_accuracy` (accuracy considering only the last element of the sequence, it is useful for being sure special end-of-sequence tokens are generated or tagged), `edit_distance` (the levenshtein distance between the predicted and ground truth sequence), `perplexity` (the perplexity of the ground truth sequence according to the model) and the `loss` itself.
You can set either of them as `validation_measure` in the `training` section of the configuration if you set the `validation_field` to be the name of a sequence feature.

Text Features
-------------

### Text Features Preprocessing

Text features are treated in the same way of sequence features, with a couple differences.
Two different tokenizations happen, one that splits at every character and one that splits on whitespace and punctuation are used, and two different keys are added to the HDF5 file, one containing the matrix of characters and one containing the matrix of words.
The same thing happens in the JSON file, which contains dictionaries for mapping characters to integers (and the inverse) and words to integers (and their inverse).
In the configuration you are able to specify which level of representation to use, if the character level or the word level.

The parameters available for preprocessing are:

- `char_tokenizer` (default `characters`): defines how to map from the raw string content of the dataset column to a sequence of characters. The default value and only available option is `characters` and the behavior is to split the string at each character.
- `char_vocab_file` (default `null`):
- `char_sequence_length_limit` (default `1024`): the maximum length of the text in characters. Texts that are longer than this value will be truncated, while sequences that are shorter will be padded.
- `char_most_common` (default `70`): the maximum number of most common characters to be considered. if the data contains more than this amount, the most infrequent characters will be treated as unknown.
- `word_tokenizer` (default `space_punct`): defines how to map from the raw string content of the dataset column to a sequence of elements. For the available options refer to the [Tokenizers](#tokenizers)section.
- `pretrained_model_name_or_path` (default `null`):
- `word_vocab_file` (default `null`):
- `word_sequence_length_limit` (default `256`): the maximum length of the text in words. Texts that are longer than this value will be truncated, while texts that are shorter will be padded.
- `word_most_common` (default `20000`): the maximum number of most common words to be considered. If the data contains more than this amount, the most infrequent words will be treated as unknown.
- `padding_symbol` (default `<PAD>`): the string used as a padding symbol. Is is mapped to the integer ID 0 in the vocabulary.
- `unknown_symbol` (default `<UNK>`): the string used as a unknown symbol. Is is mapped to the integer ID 1 in the vocabulary.
- `padding` (default `right`): the direction of the padding. `right` and `left` are available options.
- `lowercase` (default `false`): if the string has to be lowercased before being handled by the tokenizer.
- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `""`): the value to replace the missing values with in case the `missing_value_strategy` is `fill-value`.

Example of text preprocessing.

```yaml
name: text_column_name
type: text
level: word
preprocessing:
    char_tokenizer: characters
    char_vocab_file: null
    char_sequence_length_limit: 1024
    char_most_common: 70
    word_tokenizer: space_punct
    pretrained_model_name_or_path: null
    word_vocab_file: null
    word_sequence_length_limit: 256
    word_most_common: 20000
    padding_symbol: <PAD>
    unknown_symbol: <UNK>
    padding: right
    lowercase: false
    missing_value_strategy: fill_with_const
    fill_value: ""
```

### Text Input Features and Encoders

Text input feature parameters are

- `encoder` (default `parallel_cnn`): encoder to use for the input text feature. The available encoders come from [Sequence Features](#sequence-input-features-and-encoders) and these text specific encoders: `bert`, `gpt`, `gpt2`, `xlnet`, `xlm`, `roberta`, `distilbert`, `ctrl`, `camembert`, `albert`, `t5`, `xlmroberta`, `flaubert`, `electra`, `longformer` and `auto-transformer`.
- `level` (default `word`): `word` specifies using text words, `char` use individual characters.
- `tied_weights` (default `null`): name of the input feature to tie the weights of the encoder with. It needs to be the name of a feature of the same type and with the same encoder parameters.


#### BERT Encoder

The `bert` encoder loads a pretrained [BERT](https://arxiv.org/abs/1810.04805) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `bert-base-uncased`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/bert.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### GPT Encoder

The `gpt` encoder loads a pretrained [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `openai-gpt`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/gpt.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### GPT-2 Encoder 

The `gpt2` encoder loads a pretrained [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `gpt2`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/gpt2.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### XLNet Encoder 

The `xlnet` encoder loads a pretrained [XLNet](https://arxiv.org/abs/1906.08237) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `xlnet-base-cased`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/xlnet.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### XLM Encoder 

The `xlm` encoder loads a pretrained [XLM](https://arxiv.org/abs/1901.07291) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `xlm-mlm-en-2048`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/xlm.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### RoBERTa Encoder 

The `roberta` encoder loads a pretrained [RoBERTa](https://arxiv.org/abs/1907.11692) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `roberta-base`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/roberta.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### DistilBERT Encoder 

The `distilbert` encoder loads a pretrained [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `istilbert-base-uncased`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/distilbert.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### CTRL Encoder 

The `ctrl` encoder loads a pretrained [CTRL](https://arxiv.org/abs/1909.05858) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `ctrl`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/ctrl.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### CamemBERT Encoder 

The `camembert` encoder loads a pretrained [CamemBERT](https://arxiv.org/abs/1911.03894) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `jplu/tf-camembert-base`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/camembert.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### ALBERT Encoder 

The `albert` encoder loads a pretrained [ALBERT](https://arxiv.org/abs/1909.11942) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `albert-base-v2`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/albert.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### T5 Encoder 

The `t5` encoder loads a pretrained [T5](https://arxiv.org/pdf/1910.10683.pdf) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `t5-small`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/t5.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### XLM-RoBERTa Encoder 

The `xlmroberta` encoder loads a pretrained [XLM-RoBERTa](https://arxiv.org/abs/1911.02116) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `jplu/tf-xlm-reoberta-base`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/xlmroberta.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### FlauBERT Encoder 

The `flaubert` encoder loads a pretrained [FlauBERT](https://arxiv.org/abs/1912.05372) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `jplu/tf-flaubert-base-uncased`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/flaubert.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### ELECTRA Encoder 

The `electra` encoder loads a pretrained [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `google/electra-small-discriminator`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/electra.html).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### Longformer Encoder 

The `longformer` encoder loads a pretrained [Longformer](https://arxiv.org/pdf/2004.05150.pdf) model using the Hugging Face transformers package.

- `pretrained_model_name_or_path` (default `allenai/longformer-base-4096`): it can be either the name of a model or a path where it was downloaded. For details on the variants available refer to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/longformer.html).
- `reduced_output` (default `cls_pooled`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `cls_pool`, `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### Auto-Transformer Encoder

The `auto_transformer` encoder loads a pretrained model using the Hugging Face transformers package.
It's the best option for customly trained models that don't fit into the other pretrained transformers encoders.

- `pretrained_model_name_or_path`: it can be either the name of a model or a path where it was downloaded. For details on the available models to the [Hugging Face documentation](https://huggingface.co/transformers/model_doc/auto.html#tfautomodel).
- `reduced_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).
- `trainable` (default `false`): if `true` the weights of the encoder will be trained, otherwise they will be kept frozen. 


#### Example usage

Example text input feature encoder usage:

```yaml
name: text_column_name
type: text
level: word
encoder: bert
tied_weights: null
pretrained_model_name_or_path: bert-base-uncased
reduced_output: cls_pooled
trainable: false
```

### Text Output Features and Decoders

The decoders are the same used for the [Sequence Features](#sequence-output-features-and-decoders).
The only difference is that you can specify an additional `level` parameter with possible values `word` or `char` to force to use the text words or characters as inputs (by default the encoder will use `word`).

Example text input feature using default values:

```yaml
name: sequence_column_name
type: text
level: word
decoder: generator
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: softmax_cross_entropy
    confidence_penalty: 0
    robust_lambda: 0
    class_weights: 1
    class_similarities: null
    class_similarities_temperature: 0
    labels_smoothing: 0
    negative_samples: 0
    sampler: null
    distortion: 1
    unique: false
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
cell_type: rnn
state_size: 256
embedding_size: 256
beam_width: 1
attention: null
tied_embeddings: null
max_sequence_length: 0
```

### Text Features Measures

The measures are the same used for the [Sequence Features](#sequence-features-measures).


Time Series Features
--------------------

### Time Series Features Preprocessing

Timeseries features are treated in the same way of sequence features, with the only difference being that the matrix in the HDF5 file does not have integer values, but float values.
Moreover, there is no need for any mapping in the JSON file.

### Time Series Input Features and Encoders

The encoders are the same used for the [Sequence Features](#sequence-input-features-and-encoders).
The only difference is that time series features don't have an embedding layer at the beginning, so the `b x s` placeholders (where `b` is the batch size and `s` is the sequence length) are directly mapped to a `b x s x 1` tensor and then passed to the different sequential encoders.

### Time Series Output Features and Decoders

There are no time series decoders at the moment (WIP), so time series cannot be used as output features.

### Time Series Features Measures

As no time series decoders are available at the moment, there are also no time series measures.

Audio Features
--------------

### Audio Features Preprocessing

Ludwig supports reads in audio files using Python's library [SoundFile](https://pypi.org/project/SoundFile/) therefore supporting WAV, FLAC, OGG and MAT files.

- `audio_file_length_limit_in_s`: (default `7.5`): float value that defines the maximum limit of the audio file in seconds. All files longer than this limit are cut off. All files shorter than this limit are padded with `padding_value`
- `missing_value_strategy` (default: `backfill`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `in_memory` (default `true`): defines whether image dataset will reside in memory during the training process or will be dynamically fetched from disk (useful for large datasets). In the latter case a training batch of input images will be fetched from disk each training iteration. At the moment only `in_memory` = true is supported.
- `padding_value`: (default 0): float value that is used for padding. 
- `norm`: (default `null`) the normalization method that can be used for the input data. Supported methods: `null` (data is not normalized), `per_file` (z-norm is applied on a “per file” level)
- `audio_feature`: (default `{ type: raw }`) dictionary that takes as input the audio feature `type` as well as additional parameters if `type != raw`. The following parameters can/should be defined in the dictionary:
	- `type` (default `raw`): defines the type of audio features to be used. Supported types at the moment are `raw`, `stft`, `stft_phase`, `group_delay`. For more detail, check [Audio Input Features and Encoders](#audio-input-features-and-encoders).
	- `window_length_in_s`: defines the window length used for the short time Fourier transformation (only needed if `type != raw`).
	- `window_shift_in_s`: defines the window shift used for the short time Fourier transformation (also called hop_length) (only needed if `type != raw`).
	- `num_fft_points`: (default `window_length_in_s * sample_rate` of audio file) defines the number of fft points used for the short time Fourier transformation. If `num_fft_points > window_length_in_s * sample_rate`, then the signal is zero-padded at the end. `num_fft_points` has to be `>= window_length_in_s * sample_rate` (only needed if `type != raw`).
	- `window_type`: (default `hamming`): defines the type window the signal is weighted before the short time Fourier transformation. All windows provided by [scipy’s window function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html) can be used (only needed if `type != raw`).
	- `num_filter_bands`: defines the number of filters used in the filterbank (only needed if `type == fbank`).


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

### Audio Input Features and Encoders

Audio files are transformed into one of the following types according to `type` in `audio_feature` in `preprocessing`.

- `raw`: audio file is transformed into a float valued tensor of size `N x L x W` (where `N` is the size of the dataset and `L` corresponds to `audio_file_length_limit_in_s * sample_rate` and `W = 1`).
- `stft`: audio is transformed to the `stft` magnitude. Audio file is transformed into a float valued tensor of size `N x L x W` (where `N` is the size of the dataset, `L` corresponds to `ceil(audio_file_length_limit_in_s * sample_rate - window_length_in_s * sample_rate + 1/ window_shift_in_s * sample_rate) + 1` and `W` corresponds to `num_fft_points / 2`).
- `fbank`: audio file is transformed to FBANK features (also called log Mel-filter bank values). FBANK features are implemented according to their definition in the [HTK Book](http://www.inf.u-szeged.hu/~tothl/speech/htkbook.pdf): Raw Signal -> Preemphasis -> DC mean removal -> `stft` magnitude -> Power spectrum: `stft^2` -> mel-filter bank values: triangular filters equally spaced on a Mel-scale are applied -> log-compression: `log()`. Overall the audio file is transformed into a float valued tensor of size `N x L x W` with `N,L` being equal to the ones in `stft` and `W` being equal to `num_filter_bands`.
- `stft_phase`: the phase information for each stft bin is appended to the `stft` magnitude so that the audio file is transformed into a float valued tensor of size `N x L x 2W` with `N,L,W` being equal to the ones in `stft`.
- `group_delay`: audio is transformed to group delay features according to Equation (23) in this [paper](https://www.ias.ac.in/article/fullyext/sadh/036/05/0745-0782). Group_delay features has the same tensor size as `stft`.

The encoders are the same used for the [Sequence Features](#sequence-input-features-and-encoders).
The only difference is that time series features don't have an embedding layer at the beginning, so the `b x s` placeholders (where `b` is the batch size and `s` is the sequence length) are directly mapped to a `b x s x w` (where `w` is `W` as described above) tensor and then passed to the different sequential encoders.


### Audio Output Features and Decoders

There are no audio decoders at the moment (WIP), so audio cannot be used as output features.

### Audio Features Measures

As no audio decoders are available at the moment, there are also no audio measures.


Image Features
--------------

### Image Features Preprocessing

Ludwig supports both grayscale and color images.
The number of channels is inferred, but make sure all your images have the same number of channels.
During preprocessing, raw image files are transformed into numpy ndarrays and saved in the hdf5 format.
All images in the dataset should have the same size.
If they have different sizes, a `resize_method`, together with a target `width` and `height`, must be specified in the feature preprocessing parameters.

- `missing_value_strategy` (default: `backfill`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `in_memory` (default `true`): defines whether image dataset will reside in memory during the training process or will be dynamically fetched from disk (useful for large datasets). In the latter case a training batch of input images will be fetched from disk each training iteration.
- `num_processes` (default 1): specifies the number of processes to run for preprocessing images.
- `resize_method` (default `crop_or_pad`): available options: `crop_or_pad` - crops images larger than the specified `width` and `height` to the desired size or pads smalled images using edge padding; `interpolate` - uses interpolation to resize images to the specified `width` and `height`.
- `height` (default `null`): image height in pixels, must be set if resizing is required
- `width` (default `null`): image width in pixels, must be set if resizing is required
- `num_channels` (default `null`): number of channels in the images. By default, if the value is `null`, the number of channels of the first image of the dataset will be used and if there is an image in the dataset with a different number of channels, an error will be reported. If the value specified is not `null`, images in the dataset will be adapted to the specified size. If the value is `1`, all images with more then one channel will be greyscaled and reduced to one channel (trasparecy will be lost). If the value is `3` all images with 1 channel will be repeated 3 times to obtain 3 channels, while images with 4 channels will lose the transparecy channel. If the value is `4`, all the images with less than 4 channels will have the remaining channels filled with zeros.
- `scaling` (default `pixel_normalization`): what scaling to perform on images. By default `pixel_normalization` is performed, which consists in dividing each pixel values by 255, but `pixel_standardization` is also available, whic uses [TensorFlow's per image standardization](https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization).

Depending on the application, it is preferrable not to exceed a size of `256 x 256`, as bigger sizes will, in most cases, not provide much advantage in terms of performance, while they will considerably slow down training and inference and also make both forward and backward passes consume considerably more memory, leading to memory overflows on machines with limited amounts of RAM or on GPUs with limited amounts of VRAM.

Example of a preprocessing specification:

```yaml
name: image_feature_name
type: image
preprocessing:
  height: 128
  width: 128
  resize_method: interpolate
  scaling: pixel_normalization
```


### Image Input Features and Encoders

Input image features are transformed into a float valued tensors of size `N x H x W x C` (where `N` is the size of the dataset and `H x W` is a specific resizing of the image that can be set, and `C` is the number of channels) and added to HDF5 with a key that reflects the name of column in the dataset.
The column name is added to the JSON file, with an associated dictionary containing preprocessing information about the sizes of the resizing.

Currently there are two encoders supported for images: Convolutional Stack Encoder and ResNet encoder which can be set by setting `encoder` parameter to `stacked_cnn` or `resnet` in the input feature dictionary in the configuration (`stacked_cnn` is the default one).



#### Convolutional Stack Encoder

Convolutional Stack Encoder takes the following optional parameters:

- `conv_layers` (default `null`): it is a list of dictionaries containing the parameters of all the convolutional layers. The length of the list determines the number of stacked convolutional layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `filter_size`, `num_filters`, `pool_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `conv_layers` and `num_conv_layers` are `null`, a default list will be assigned to `conv_layers` with the value `[{filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 7, pool_size: 3, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: false}, {filter_size: 3, pool_size: null, regularize: true}, {filter_size: 3, pool_size: 3, regularize: true}]`.
- `num_conv_layers` (default `null`): if `conv_layers` is `null`, this is the number of stacked convolutional layers.
- `filter_size` (default `3`): if a `filter_size` is not already specified in `conv_layers` this is the default `filter_size` that will be used for each layer. It indicates how wide is the 1d convolutional filter.
- `num_filters` (default `256`): if a `num_filters` is not already specified in `conv_layers` this is the default `num_filters` that will be used for each layer. It indicates the number of filters, and by consequence the output channels of the 2d convolution.
- `strides` (default `(1, 1)`): specifying the strides of the convolution along the height and width
- `padding` (default `valid`): one of `valid` or `same`.
- `dilation_rate` (default `(1, 1)`): specifying the dilation rate to use for dilated convolution.
- `conv_use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `conv_weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `conv_bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `conv_bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `conv_activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `conv_norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `conv_norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `conv_activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `conv_dropout` (default `0`): dropout rate
- `pool_function` (default `max`):  pooling function: `max` will select the maximum value.  Any of these--`average`, `avg` or `mean`--will compute the mean value.
- `pool_size` (default `(2, 2)`): if a `pool_size` is not already specified in `conv_layers` this is the default `pool_size` that will be used for each layer. It indicates the size of the max pooling that will be performed along the `s` sequence dimension after the convolution operation.
- `pool_strides` (default `null`): factor to scale down
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `1`): This is the number of stacked fully connected layers.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `fc_use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `fc_weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `fc_bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `fc_weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `fc_bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `fc_activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `fc_norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `fc_norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `fc_activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `fc_dropout` (default `0`): dropout rate

Example image feature entry using a convolutional stack encoder (with default parameters) in the input features list:

```yaml
name: image_column_name
type: image
encoder: stacked_cnn
tied_weights: null
conv_layers: null
num_conv_layers: null
filter_size: 3
num_filters: 256
strides: (1, 1)
padding: valid
dilation_rate: (1, 1)
conv_use_bias: true
conv_weights_initializer: glorot_uniform
conv_bias_initializer: zeros
weights_regularizer: null
conv_bias_regularizer: null
conv_activity_regularizer: null
conv_norm: null
conv_norm_params: null
conv_activation: relu
conv_dropout: 0
pool_function: max
pool_size: (2, 2)
pool_strides: null
fc_layers: null
num_fc_layers: 1
fc_size: 256
fc_use_bias: true
fc_weights_initializer: glorot_uniform
fc_bias_initializer: zeros
fc_weights_regularizer: null
fc_bias_regularizer: null
fc_activity_regularizer: null
fc_norm: null
fc_norm_params: null
fc_activation: relu
fc_dropout: 0
preprocessing:  # example pre-processing
    height: 28
    width: 28
    num_channels: 1

```

#### ResNet Encoder

[ResNet](https://arxiv.org/abs/1603.05027) Encoder takes the following optional parameters:

- `resnet_size` (default `50`): A single integer for the size of the ResNet model. If has to be one of the following values: `8`, `14`, `18`, `34`, `50`, `101`, `152`, `200`.
- `num_filters` (default `16`): It indicates the number of filters, and by consequence the output channels of the 2d convolution.
- `kernel_size` (default `3`): The kernel size to use for convolution.
- `conv_stride` (default `1`): Stride size for the initial convolutional layer.
- `first_pool_size` (default `null`): Pool size to be used for the first pooling layer. If none, the first pooling layer is skipped.
- `batch_norm_momentum` (default `0.9`): Momentum of the batch norm running statistics. The suggested parameter in [TensorFlow's implementation](https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py#L36) is `0.997`, but that leads to a big discrepancy between the normalization at training time and test time, so the default value is a more conservative `0.9`.
- `batch_norm_epsilon` (default `0.001`): Epsilon of the batch norm. The suggested parameter in [TensorFlow's implementation](https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py#L37) is `1e-5`, but that leads to a big discrepancy between the normalization at training time and test time, so the default value is a more conservative `0.001`.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `1`): This is the number of stacked fully connected layers.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

Example image feature entry using a ResNet encoder (with default parameters) in the input features list:

```yaml
name: image_column_name
type: image
encoder: resnet
tied_weights: null
resnet_size: 50
num_filters: 16
kernel_size: 3
conv_stride: 1
first_pool_size: null
batch_norm_momentum: 0.9
batch_norm_epsilon: 0.001
fc_layers: null
num_fc_layers: 1
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
preprocessing:
    height: 224
    width: 224
    num_channels: 3
```

### Image Output Features and Decoders

There are no image decoders at the moment (WIP), so image cannot be used as output features.

### Image Features Measures

As no image decoders are available at the moment, there are also no image measures.


Date Features
-------------

### Date Features Preprocessing

Ludwig will try to infer the date format automatically, but a specific fomrat can be provided. 
The format is the same one described in the [datetime package documentation](https://docs.python.org/2/library/time.html#time.strptime).

- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `""`): the value to replace the missing values with in case the `missing_value_strategy` is `fill_value`. This can be a datetime string, if left empty the current datetime will be used.
- `datetime_format` (default `null`): this parameter can be either `null`, which implies the datetime format is inferred automaticall, or a datetime format string.

Example of a preprocessing specification:

```yaml
name: date_feature_name
type: date
preprocessing:
  missing_value_strategy: fill_with_const
  fill_value: ''
  datetime_format: "%d %b %Y"
```


### Date Input Features and Encoders

Input date features are transformed into a int valued tensors of size `N x 8` (where `N` is the size of the dataset and the 8 dimensions contain year, month, day, weekday, yearday, hour, minute and second) and added to HDF5 with a key that reflects the name of column in the dataset.

Currently there are two encoders supported for dates: Embed Encoder and Wave encoder which can be set by setting `encoder` parameter to `embed` or `wave` in the input feature dictionary in the configuration (`embed` is the default one).

#### Embed Encoder

This encoder passes the year through a fully connected layer of one neuron and embeds all other elements for the date, concatenates them and passes the concatenated representation through fully connected layers.
It takes the following optional parameters:

- `embedding_size` (default `10`): it is the maximum embedding size adopted..
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `dropout` (default `false`): determines if there should be a dropout layer before returning the encoder output.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `0`): This is the number of stacked fully connected layers.
- `fc_size` (default `10`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

Example date feature entry in the input features list using an embed encoder:

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
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
```

#### Wave Encoder

This encoder passes the year through a fully connected layer of one neuron and represents all other elements for the date by taking the sine of their value with a different period (12 for months, 31 for days, etc.), concatenates them and passes the concatenated representation through fully connected layers.
It takes the following optional parameters:

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_fc_layers` (default `0`): This is the number of stacked fully connected layers.
- `fc_size` (default `10`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

Example date feature entry in the input features list using a wave encoder:

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
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
```

### Date Output Features and Decoders

There are no date decoders at the moment (WIP), so date cannot be used as output features.

### Date Features Measures

As no date decoders are available at the moment, there are also no date measures.


H3 Features
-------------

H3 is a indexing system for representing geospatial data.
For more details about it refer to: https://eng.uber.com/h3/ .

### H3 Features Preprocessing

Ludwig will parse the H3 64bit encoded format automatically. 
The parameters for preprocessing are:

- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value in a binary column. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default `576495936675512319`): the value to replace the missing values with in case the `missing_value_strategy` is `fill_value`. This is a 64bit integer comaptible with the H3 bit layout. The default value encodes mode 1, edge 0, resolution 0, base_cell 0.

Example of a preprocessing specification:

```yaml
name: h3_feature_name
type: h3
preprocessing:
  missing_value_strategy: fill_with_const
  fill_value: 576495936675512319
```


### H3 Input Features and Encoders

Input date features are transformed into a int valued tensors of size `N x 8` (where `N` is the size of the dataset and the 8 dimensions contain year, month, day, weekday, yearday, hour, minute and second) and added to HDF5 with a key that reflects the name of column in the dataset.

Currently there are two encoders supported for dates: Embed Encoder and Wave encoder which can be set by setting `encoder` parameter to `embed` or `wave` in the input feature dictionary in the configuration (`embed` is the default one).

#### Embed Encoder

This encoder encodes each components of the H3 representation (mode, edge, resolution, base cell and childern cells) with embeddings.
Chidren cells with value `0` will be masked out.
After the embedding, all embeddings are summed and optionally passed through a stack of fully connected layers.
It takes the following optional parameters:

- `embedding_size` (default `10`): it is the maximum embedding size adopted..
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead.
- `num_fc_layers` (default `0`): This is the number of stacked fully connected layers.
- `fc_size` (default `10`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

Example date feature entry in the input features list using an embed encoder:

```yaml
name: h3_column_name
type: h3
encoder: embed
embedding_size: 10
embeddings_on_cpu: false
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
```

#### Weighted Sum Embed Encoder

This encoder encodes each components of the H3 representation (mode, edge, resolution, base cell and childern cells) with embeddings.
Chidren cells with value `0` will be masked out.
After the embedding, all embeddings are summed with a weighted sum (with learned weights) and optionally passed through a stack of fully connected layers.
It takes the following optional parameters:

- `embedding_size` (default `10`): it is the maximum embedding size adopted..
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `should_softmax` (default `false`): determines if the weights of the weighted sum should be passed though a softmax layer before being used.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead.
- `num_fc_layers` (default `0`): This is the number of stacked fully connected layers.
- `fc_size` (default `10`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate
- `reduce_output` (default `sum`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).


Example date feature entry in the input features list using an embed encoder:

```yaml
name: h3_column_name
type: h3
encoder: weighted_sum
embedding_size: 10
embeddings_on_cpu: false
should_softmax: false
fc_layers: null
num_fc_layers: 0
fc_size: 10
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
reduce_output: sum
```

#### RNN Encoder

This encoder encodes each components of the H3 representation (mode, edge, resolution, base cell and childern cells) with embeddings.
Chidren cells with value `0` will be masked out.
After the embedding, all embeddings are passed through an RNN encoder.
The intuition behind this is that, starting from the base cell, the sequence of children cells can be seen as a sequence encoding the path in the tree of all H3 hexes, thus the encoding with  recurrent model.
It takes the following optional parameters:

- `embedding_size` (default `10`): it is the maximum embedding size adopted..
- `embeddings_on_cpu` (default `false`): by default embeddings matrices are stored on GPU memory if a GPU is used, as it allows for faster access, but in some cases the embedding matrix may be really big and this parameter forces the placement of the embedding matrix in regular memory and the CPU is used to resolve them, slightly slowing down the process as a result of data transfer between CPU and GPU memory.
- `num_layers` (default `1`): the number of stacked recurrent layers.
- `state_size` (default `256`): the size of the state of the rnn.
- `cell_type` (default `rnn`): the type of recurrent cell to use. Available values are: `rnn`, `lstm`, `lstm_block`, `lstm`, `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`. For reference about the differences between the cells please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell). We suggest to use the `block` variants on CPU and the `cudnn` variants on GPU because of their increased speed.
- `bidirectional` (default `false`): if `true` two recurrent networks will perform encoding in the forward and backward direction and their outputs will be concatenated.
- `activation` (default `'tanh'`): activation function to use
- `recurrent_activation` (default `'sigmoid'`): activation function to use in the recurrent step
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `unit_forget_bias` (default `true`): If `true`, add 1 to the bias of the forget gate at initialization
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `recurrent_initializer` (default `'orthogonal'`): initializer for recurrent matrix weights
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `recurrent_regularizer` (default `null`): regularizer function applied to recurrent matrix weights
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `dropout` (default `0.0`): dropout rate
- `recurrent_dropout` (default `0.0`): dropout rate for recurrent state
- `initializer` (default `null`): the initializer to use. If `null`, the default initialized of each variable is used (`glorot_uniform` in most cases). Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `regularize` (default `true`): if `true` the embedding weights are added to the set of weights that get regularized by a regularization loss (if the `regularization_lambda` in `training` is greater than 0).
- `reduce_output` (default `last`): defines how to reduce the output tensor along the `s` sequence length dimension if the rank of the tensor is greater than 2. Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension) and  `null` (which does not reduce and returns the full tensor).

Example date feature entry in the input features list using an embed encoder:

```yaml
name: h3_column_name
type: h3
encoder: rnn
embedding_size: 10
embeddings_on_cpu: false
num_layers: 1
cell_type: rnn
state_size: 10
bidirectional: false
activation: tanh
recurrent_activation: sigmoid
use_bias: true
unit_forget_bias: true
weights_initializer: glorot_uniform
recurrent_initializer: orthogonal
bias_initializer: zeros
weights_regularizer: null
recurrent_regularizer: null
bias_regularizer: null
activity_regularizer: null
dropout: 0.0
recurrent_dropout: 0.0
initializer: null
regularize: true
reduce_output: last
```

### H3 Output Features and Decoders

There are no date decoders at the moment (WIP), so H3 cannot be used as output features.

### H3 Features Measures

As no H3 decoders are available at the moment, there are also no date measures.

Vector Features
---------------

Vector features allow to provide an ordered set of numerical values all at once.
This is useful for providing pre-trained representations or activations obtained from other models or for providing multivariate inputs and outputs.
An interesting use of vector features is the possibility to provide a probability distribution as output for a multiclass classification problem instead of just the correct class like it is possible to do with category features.
This is useful for distillation and noise-aware losses.

### Vector Feature Preprocessing

The data is expected as whitespace separated numerical values. Example: "1.0 0.0 1.04 10.49".  All vectors are expected to be of the same size.

Preprocessing parameters:

- `vector_size` (default `null`): size of the vector. If not provided, it will be inferred from the data.
- `missing_value_strategy` (default `fill_with_const`): what strategy to follow when there's a missing value. The value should be one of `fill_with_const` (replaces the missing value with a specific value specified with the `fill_value` parameter), `fill_with_mode` (replaces the missing values with the most frequent value in the column), `fill_with_mean` (replaces the missing values with the mean of the values in the column), `backfill` (replaces the missing values with the next valid value).
- `fill_value` (default ""): the value to replace the missing values with in case the `missing_value_strategy` is `fill_value`. 

### Vector Feature Encoders

The vector feature supports two encoders: `dense` and `passthrough`.  Only the `dense` encoder has additional parameters, which is shown next.

#### Dense Encoder

For vector features, you can use a dense encoder (stack of fully connected layers).
It takes the following parameters:

- `layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the encoder will be used instead. If both `fc_layers` and `num_fc_layers` are `null`, a default list will be assigned to `fc_layers` with the value `[{fc_size: 512}, {fc_size: 256}]` (only applies if `reduce_output` is not `null`).
- `num_layers` (default `0`): This is the number of stacked fully connected layers.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate

Example vector feature entry in the input features list using an dense encoder:

```yaml
name: vector_column_name
type: vector
encoder: dense
layers: null
num_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
```

### Vector Feature Decoders

Vector features can be used when multi-class classification needs to be performed with a noise-aware loss or when the task is multivariate regression.
There is only one decoder available for set features and it is a (potentially empty) stack of fully connected layers, followed by a projection into a vector of size (optionally followed by a softmax in the case of multi-class classification).

```
+--------------+   +---------+   +-----------+
|Combiner      |   |Fully    |   |Projection |   +------------------+
|Output        +--->Connected+--->into Output+--->Softmax (optional)|
|Representation|   |Layers   |   |Space      |   +------------------+
+--------------+   +---------+   +-----------+
```

These are the available parameters of the set output feature

- `reduce_input` (default `sum`): defines how to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `dependencies` (default `[]`): the output features this one is dependent on. For a detailed explanation refer to [Output Features Dependencies](#output-features-dependencies).
- `reduce_dependencies` (default `sum`): defines how to reduce the output of a dependent feature that is not a vector, but a matrix or a higher order tensor, on the first dimension (second if you count the batch dimension). Available values are: `sum`, `mean` or `avg`, `max`, `concat` (concatenates along the first dimension), `last` (returns the last vector of the first dimension).
- `softmax` (default `false`): determines if to apply a softmax at the end of the decoder. It is useful for predicting a vector of values that sum up to 1 and can be interpreted as probabilities.
- `loss` (default `{type: mean_squared_error}`): is a dictionary containing a loss `type`. The available loss `type` are `mean_squared_error`, `mean_absolute_error` and `softmax_cross_entropy` (use it only if `softmax` is `true`).

These are the available parameters of a set output feature decoder

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `clip` (default `null`): If not `null` it specifies a minimum and maximum value the predictions will be clipped to. The value can be either a list or a tuple of length 2, with the first value representing the minimum and the second the maximum. For instance `(-5,5)` will make it so that all predictions will be clipped in the `[-5,5]` interval.

Example vector feature entry (with default parameters) in the output features list:

```yaml
name: vector_column_name
type: vector
reduce_input: sum
dependencies: []
reduce_dependencies: sum
loss:
    type: sigmoid_cross_entropy
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
activation: relu
clip: null
```

### Vector Features Measures

The measures that are calculated every epoch and are available for numerical features are `mean_squared_error`, `mean_absolute_error`, `r2` and the `loss` itself.
You can set either of them as `validation_measure` in the `training` section of the configuration if you set the `validation_field` to be the name of a numerical feature.


Combiners
---------

Combiners are the part of the model that take the outputs of the encoders of all input features and combine them before providing the combined representation to the different output decoders.
If you don't specify a combiner, the `concat` combiner will be used.

### Concat Combiner

The `concat` combiner assumes all outputs from encoders are tensors of size `b x h` where `b` is the batch size and `h` is the hidden dimension, which can be different for each input.
If inputs are tensors with different shapes, set the `flatten_inputs` parameter to `true`.
It concatenates along the `h` dimension, and then (optionally) passes the concatenated tensor through a stack of fully connected layers.
It returns the final `b x h'` tensor where `h'` is the size of the last fully connected layer or the sum of the sizes of the `h` of all inputs in the case there are no fully connected layers.
If there's only one input feature and no fully connected layers are specified, the output of the input feature is just passed through as output.

```
+-----------+
|Input      |
|Feature 1  +-+
+-----------+ |            +---------+
+-----------+ | +------+   |Fully    |
|...        +--->Concat+--->Connected+->
+-----------+ | +------+   |Layers   |
+-----------+ |            +---------+
|Input      +-+
|Feature N  |
+-----------+
```

These are the available parameters of a `concat` combiner:

- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate.
- `flatten_inputs` (default `false`): if `true` flatten the tensors from all the input features into a vector.
- `residual` (default `false`): if `true` adds a residual connection to each fully connected layer block. It is required that all fully connected layers have the same size for this parameter to work correctly.

Example configuration of a `concat` combiner:

```yaml
type: concat
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: 'glorot_uniform'
bias_initializer: 'zeros'
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
flatten_inputs: false
residual: false
```

### Sequence Concat Combiner

The `sequence?concat` combiner assumes at least one output from encoders is a tensors of size `b x s x h` where `b` is the batch size, `s` is the length of the sequence and `h` is the hidden dimension.
The sequence / text / sequential input can be specified with the `main_sequence_feature` parameter that should have the name of the sequential feature as value.
If no `main_sequence_feature` is specified, the combiner will look through all the features in the order they are defined in the configuration and will look for a feature with a rank 3 tensor output (sequence, text or time series).
If it cannot find one it will raise an exception, otherwise the output of that feature will be used for concatenating the other features along the sequence `s` dimension.

If there are other input features with a rank 3 output tensor, the combiner will concatenate them alongside the `s` dimension, which means that all of them must have identical `s` dimension, otherwise an error will be thrown.
Specifically, as the placeholders of the sequential features are of dimension `[None, None]` in order to make the `BucketedBatcher` trim longer sequences to their actual length, the check if the sequences are of the same length cannot be performed at model building time, and a dimension mismatch error will be returned during training when a datapoint with two sequential features of different lengths are provided.

Other features that have a `b x h` rank 2 tensor output will be replicated `s` times and concatenated to the `s` dimension.
The final output is a `b x s x h'` tensor where `h'` is the size of the concatenation of the `h` dimensions of all input features.

```
Sequence
Feature
Output

+---------+
|emb seq 1|
+---------+
|...      +--+
+---------+  |  +-----------------+
|emb seq n|  |  |emb seq 1|emb oth|   +------+
+---------+  |  +-----------------+   |      |
             +-->...      |...    +-->+Reduce+->
Other        |  +-----------------+   |      |
Feature      |  |emb seq n|emb oth|   +------+
Output       |  +-----------------+
             |
+-------+    |
|emb oth+----+
+-------+
```

These are the available parameters of a `sequence_concat` combiner:

- `main_sequence_feature` (default `null`): name of the sequence / text/ time series feature to concatenate the outputs of the other features to. If no `main_sequence_feature` is specified, the combiner will look through all the features in the order they are defined in the configuration and will look for a feature with a rank 3 tensor output (sequence, text or time series). If it cannot find one it will raise an exception, otherwise the output of that feature will be used for concatenating the other features along the sequence `s` dimension. If there are other input features with a rank 3 output tensor, the combiner will concatenate them alongside the `s` dimension, which means that all of them must have identical `s` dimension, otherwise an error will be thrown.
- `reduce_output` (default `null`): describes the strategy to use to aggregate the embeddings of the items of the set. Possible values are `null`, `sum`, `mean` and `sqrt` (the weighted sum divided by the square root of the sum of the squares of the weights).

Example configuration of a `sequence_concat` combiner:

```yaml
type: sequence_concat
main_sequence_feature: null
reduce_output: null
```

### Sequence Combiner

The `sequence` combiner stacks a sequence concat combiner with a sequence encoder one on top of each other.
All the considerations about inputs tensor ranks describer for the [sequence concat combiner](#sequence-concat-combiner) apply also in this case, but the main difference is that this combiner uses the `b x s x h'` output of the sequence concat combiner, where `b` is the batch size, `s` is the sequence length and `h'` is the sum of the hidden dimensions of all input features, as input for any of the sequence encoders described in the [sequence features encoders section](#sequence-inpit-features-and-encoders).
Refer to that section for more detailed information about the sequence encoders and their parameters.
Also all the considerations on the shape of the outputs done for the sequence encoders apply in this case too.

```
Sequence
Feature
Output

+---------+
|emb seq 1|
+---------+
|...      +--+
+---------+  |  +-----------------+
|emb seq n|  |  |emb seq 1|emb oth|   +--------+
+---------+  |  +-----------------+   |Sequence|
             +-->...      |...    +-->+Encoder +->
Other        |  +-----------------+   |        |
Feature      |  |emb seq n|emb oth|   +--------+
Output       |  +-----------------+
             |
+-------+    |
|emb oth+----+
+-------+
```

Example configuration of a `sequence` combiner:

```yaml
type: sequence
main_sequence_feature: null
encoder: parallel_cnn
... encoder parameters ...
```

### TabNet Combiner

The `tabnet` combiner implements the [TabNet](https://arxiv.org/abs/1908.07442) model, which uses attention and sparsity to achieve high performnce on tabular data.
It assumes all outputs from encoders are tensors of size `b x h` where `b` is the batch size and `h` is the hidden dimension, which can be different for each input.
If the input tensors have a different shape, it automatically flattens them.
It returns the final `b x h'` tensor where `h'` is the user-specified output size.

```
+-----------+
|Input      |
|Feature 1  +-+
+-----------+ |            
+-----------+ | +------+   
|...        +--->TabNet+-->
+-----------+ | +------+   
+-----------+ |            
|Input      +-+
|Feature N  |
+-----------+
```

These are the available parameters of a `tabnet` combiner:

- `size`: the size of the hidden layers. `N_a` in the paper.
- `output_size`: the size of the output of each step and of the final aggregated representation. `N_d` in the paper.
- `num_steps` (default `1`): number of steps / repetitions of the the attentive transformer and feature transformer computations. `N_steps` in the paper.
- `num_total_blocks` (default `4`): total number of feature transformer block at each step.
- `num_shared_blocks` (default `2`): number of shared feature transformer blocks across the steps.
- `relaxation_factor` (default `1.5`): Factor that influences how many times a feature should be used across the steps of computation. a value of `1` implies it each feature should be use once, a higher value allows for multiple usages. `gamma` in the paper.
- `bn_epsilon` (default `0.001`): epsilon to be added to the batch norm denominator.
- `bn_momentum` (default `0.7`): momentum of the batch norm. `m_B` in the paper.
- `bn_virtual_bs` (default `null`): size of the virtual batch size used by ghost batch norm. If `null`, regular batch norm is used instead. `B_v` from the paper.
- `sparsity` (default `0.00001`): multiplier of the sparsity inducing loss. `lambda_sparse` in the paper.
- `dropout` (default `0`): dropout rate.

Example configuration of a `tabnet` combiner:

```yaml
type: tabnet
size: 32
ooutput_size: 32
num_steps: 5
num_total_blocks: 4
num_shared_blocks: 2
relaxation_factor: 1.5
bn_epsilon: 0.001
bn_momentum: 0.7
bn_virtual_bs: 128
sparsity: 0.00001
dropout: 0
```

### Transformer Combiner

The `transformer` combiner combines imput features using a stack of Transformer blocks (from [Attention Is All You Need](https://arxiv.org/abs/1706.03762)).
It assumes all outputs from encoders are tensors of size `b x h` where `b` is the batch size and `h` is the hidden dimension, which can be different for each input.
If the input tensors have a different shape, it automatically flattens them.
It then projects each input tensor to the same hidden / embedding size and encodes them wit ha stack of Tranformer layers.
Finally it applies an reduction to the outputs of the Transformer stack and applies optional fully connected layers. 
It returns the final `b x h'` tensor where `h'` is the size of the last fully connected layer or the hidden / embedding size , or it returns `b x n x h'` where `n` is the number of input features and `h'` is the hidden / embedding size if there's no reduction applied.

```
+-----------+
|Input      |
|Feature 1  +-+
+-----------+ |            
+-----------+ |  +------------+   +------+   +----------+
|...        +--->|Transformer +-->|Reduce+-->|Fully     +->
|           | |  |Stack       |   +------+   |Connected |
+-----------+ |  +------------+              |Layers    |
+-----------+ |                              +----------+            
|Input      +-+
|Feature N  |
+-----------+
```

These are the available parameters of a `transformer` combiner:

- `num_layers` (default `1`): number of layers in the stack of transformer bloks.
- `hidden_size` (default `256`): hidden / embedding size of each transformer block.
- `num_heads` (default `8`): number of heads of each transformer block.
- `transformer_fc_size` (default `256`): size of the fully connected layers inside each transformer block.
- `dropout` (default `0`): dropout rate after the transformer.
- `fc_layers` (default `null`): it is a list of dictionaries containing the parameters of all the fully connected layers. The length of the list determines the number of stacked fully connected layers and the content of each dictionary determines the parameters for a specific layer. The available parameters for each layer are: `fc_size`, `norm`, `activation`, `dropout`, `initializer` and `regularize`. If any of those values is missing from the dictionary, the default one specified as a parameter of the decoder will be used instead.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `fc_dropout` (default `0`): dropout rate for the fully connected layers.
- `fc_residual` (default `false`): if `true` adds a residual connection to each fully connected layer block. It is required that all fully connected layers have the same size for this parameter to work correctly.
- `reduce_output` (default `mean`): describes the strategy to use to aggregate the embeddings of the items of the set. Possible values are `sum`, `mean` and `sqrt` (the weighted sum divided by the square root of the sum of the squares of the weights).


Example configuration of a `transformer` combiner:

```yaml
type: transformer
num_layers: 1
hidden_size: 256
num_heads: 8
transformer_fc_size: 256
dropout: 0.1
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: True
weights_initializer: glorot_uniform
bias_initializer: zeros
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
fc_activation: relu
fc_dropout: 0
fc_residual: null
reduce_output: mean
```

### Comparator Combiner

The `comparator` combiner compares the hidden representation of two entities definef by lists of features.
It assumes all outputs from encoders are tensors of size `b x h` where `b` is the batch size and `h` is the hidden dimension, which can be different for each input.
If the input tensors have a different shape, it automatically flattens them.
It then concatenates the representations of each entity end projects them into the same size.
Finally it compares the two entity representations by dot product, element-wise multiplication, absolute difference and bilinear product. 
It returns the final `b x h'` tensor where `h'` is the size of the concatenation of the four comparisons.

```
+-----------+
|Entity 1   |
|Input      |
|Feature 1  +-+
+-----------+ |            
+-----------+ |  +-------+   +----------+
|...        +--->|Concat +-->|FC Layers +--+
|           | |  +-------+   +----------+  |
+-----------+ |                            |
+-----------+ |                            |     
|Entity 1   +-+                            |
|Input      |                              |
|Feature N  |                              |
+-----------+                              |   +---------+
                                           +-->| Compare +->
+-----------+                              |   +---------+
|Entity 2   |                              |
|Input      |                              |
|Feature 1  +-+                            |
+-----------+ |                            |
+-----------+ |  +-------+   +----------+  |
|...        +--->|Concat +-->|FC Layers +--+
|           | |  +-------+   +----------+
+-----------+ |             
+-----------+ |                                   
|Entity 2   +-+
|Input      |
|Feature N  |
+-----------+
```

These are the available parameters of a `comparator` combiner:

- `entity_1`: list of input features that compose the first entity to compare.
- `entity_2`: list of input features that compose the second entity to compare.
- `num_fc_layers` (default 0): this is the number of stacked fully connected layers that the input to the feature passes through. Their output is projected in the feature's output space.
- `fc_size` (default `256`): if a `fc_size` is not already specified in `fc_layers` this is the default `fc_size` that will be used for each layer. It indicates the size of the output of a fully connected layer.
- `use_bias` (default `true`): boolean, whether the layer uses a bias vector.
- `weights_initializer` (default `'glorot_uniform'`): initializer for the weights matrix. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `bias_initializer` (default `'zeros'`):  initializer for the bias vector. Options are: `constant`, `identity`, `zeros`, `ones`, `orthogonal`, `normal`, `uniform`, `truncated_normal`, `variance_scaling`, `glorot_normal`, `glorot_uniform`, `xavier_normal`, `xavier_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`. Alternatively it is possible to specify a dictionary with a key `type` that identifies the type of initializer and other keys for its parameters, e.g. `{type: normal, mean: 0, stddev: 0}`. To know the parameters of each initializer, please refer to [TensorFlow's documentation](https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
- `weights_regularizer` (default `null`): regularizer function applied to the weights matrix.  Valid values are `l1`, `l2` or `l1_l2`.
- `bias_regularizer` (default `null`): regularizer function applied to the bias vector.  Valid values are `l1`, `l2` or `l1_l2`.
- `activity_regularizer` (default `null`): regurlizer function applied to the output of the layer.  Valid values are `l1`, `l2` or `l1_l2`.
- `norm` (default `null`): if a `norm` is not already specified in `fc_layers` this is the default `norm` that will be used for each layer. It indicates the norm of the output and it can be `null`, `batch` or `layer`.
- `norm_params` (default `null`): parameters used if `norm` is either `batch` or `layer`.  For information on parameters used with `batch` see [Tensorflow's documentation on batch normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) or for `layer` see [Tensorflow's documentation on layer normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization).
- `activation` (default `relu`): if an `activation` is not already specified in `fc_layers` this is the default `activation` that will be used for each layer. It indicates the activation function applied to the output.
- `dropout` (default `0`): dropout rate for the fully connected layers.

Example configuration of a `comparator` combiner:

```yaml
type: comparator
entity_1: [feature_1, feature_2]
entity_3: [feature_3, feature_4]
fc_layers: null
num_fc_layers: 0
fc_size: 256
use_bias: true
weights_initializer: 'glorot_uniform'
bias_initializer: 'zeros'
weights_regularizer: null
bias_regularizer: null
activity_regularizer: null
norm: null
norm_params: null
activation: relu
dropout: 0
```
