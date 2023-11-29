| user1                     | user2                                      |
| ------------------------- | ------------------------------------------ |
| Hello! How are you doing? | Doing well, thanks!                        |
| I got promoted today      | Congratulations!                           |
| Not doing well today      | I’m sorry, can I do something to help you? |

```
ludwig experiment \
  --dataset chitchat.csv \
  --config config.yaml
```

With `config.yaml`:

```yaml
input_features:
    -
        name: user1
        type: text
        encoder: 
            type: rnn
            cell_type: lstm
            reduce_output: null

output_features:
    -
        name: user2
        type: text
        decoder: 
            type: generator
            cell_type: lstm
            attention: bahdanau
        loss:
            type: softmax_cross_entropy

trainer:
    batch_size: 96
```
