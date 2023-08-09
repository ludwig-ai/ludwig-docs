# Tutorial

On the Ludwig medium publication you can find a three part tutorial on Sentiment Analysis with Ludwig:

- [Part I (Training models from scratch)](https://medium.com/ludwig-ai/the-complete-guide-to-sentiment-analysis-with-ludwig-part-i-65a9e6bc054e?source=friends_link&sk=420a8859340d40a8f36963bd0fa4d808)
- [Part II (Finetuning pretrained models)](https://medium.com/ludwig-ai/the-complete-guide-to-sentiment-analysis-with-ludwig-part-ii-d9f3952a06c6?source=friends_link&sk=188e650703aed70f138cc990049f051e)
- [Part III (Hyperparameter Optimization)](https://medium.com/ludwig-ai/hyperparameter-optimization-with-ludwig-6e31272e43fb?source=friends_link&sk=0bc7eac913a5c529b17e8352ae278bd8)



| review                          | sentiment |
| ------------------------------- | --------- |
| The movie was fantastic!        | positive  |
| Great acting and cinematography | positive  |
| The acting was terrible!        | negative  |

```
ludwig experiment \
  --dataset sentiment.csv \
  --config config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: review
        type: text
        encoder: 
            type: parallel_cnn

output_features:
    -
        name: sentiment
        type: category
```
