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