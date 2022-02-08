This example is inspired by the classic paper [Natural Language Processing (Almost) from Scratch](https://arxiv.org/abs/1103.0398) by Collobert et al..

| sentence                    | chunks                       | part_of_speech    | named_entities      |
| --------------------------- | ---------------------------- | ----------------- | ------------------- |
| San Francisco is very foggy | B-NP I-NP B-VP B-ADJP I-ADJP | NNP NNP VBZ RB JJ | B-Loc I-Loc O O O   |
| My dog likes eating sausage | B-NP I-NP B-VP B-VP B-NP     | PRP NN VBZ VBG NN | O O O O O           |
| Brutus Killed Julius Caesar | B-NP B-VP B-NP I-NP          | NNP VBD NNP NNP   | B-Per O B-Per I-Per |

```
ludwig experiment \
--dataset nl_data.csv \
  --config_file config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: sentence
        type: sequence
        encoder: rnn
        cell: lstm
        bidirectional: true
        reduce_output: null

output_features:
    -
        name: chunks
        type: sequence
        decoder: tagger
    -
        name: part_of_speech
        type: sequence
        decoder: tagger
    -
        name: named_entities
        type: sequence
        decoder: tagger
```
