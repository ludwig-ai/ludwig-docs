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

Below are links to further explain features of configuration.

##[Input Features](input_features.md)
##[Combiner](combiner.md)
##[Output Features](output_features.md)
##[Training](training.md)
##[Preprocessing](preprocessing.md)
##[Binary Features](binary_features.md)
##[Numerical Features](numerical_features.md)
##[Category Features](category_features.md)
##[Set Features](set_features.md)
##[Bag Features](bag_features.md)
##[Sequence Features](sequence_features.md)
##[Text Features](text_features.md)
##[Time Series Features](time_series_features.md)
##[Audio Features](audio_features.md)
##[Image Features](image_features.md)
##[Date Features](date_features.md)
##[H3 Features](h3_features.md)
##[Vector Features](vector_features.md)
##[Combiners](combiners.md)