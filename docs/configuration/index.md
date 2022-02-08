The configuration is the core of Ludwig. It is a dictionary containing the following keys:

```yaml
input_features: []
combiner: {}
output_features: []
training: {}
preprocessing: {}
```

These contain all the information needed to build and train a Ludwig model.
It mixes ease of use, by means of reasonable defaults, with flexibility, by means of detailed control over the parameters of your model.
It is provided to both `experiment` and `train` commands either as a string (`--config`) or as a file (`--config_file`).
You can provide the dictionary as a YAML file. The string or the content of the file will be parsed by PyYAML into a dictionary in memory, so any style of YAML accepted by the parser is considered to be valid, so both multiline and oneline formats are accepted.
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

Only `input_features` and `output_features` are required, the other three fields
have default values, but you are free to modify them.
