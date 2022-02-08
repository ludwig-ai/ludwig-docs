This example shows how to build a text classifier with Ludwig.
It can be performed using the [Reuters-21578](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/reuters-allcats-6.zip) dataset, in particular the version available on [CMU's Text Analytics course website](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/).
Other datasets available on the same webpage, like [OHSUMED](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/ohsumed-allcats-6.zip), is a well-known medical abstracts dataset, and [Epinions.com](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/epinions.zip), a dataset of product reviews, can be used too as the name of the columns is the same.

| text                                                                                            | class       |
| ----------------------------------------------------------------------------------------------- | ----------- |
| Toronto  Feb 26 - Standard Trustco said it expects earnings in 1987 to increase at least 15...  | earnings    |
| New York  Feb 26 - American Express Co remained silent on market rumors...                      | acquisition |
| BANGKOK  March 25 - Vietnam will resettle 300000 people on state farms known as new economic... | coffee      |

```
ludwig experiment \
  --dataset text_classification.csv \
  --config_file config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: text
        type: text
        level: word
        encoder: parallel_cnn

output_features:
    -
        name: class
        type: category
```