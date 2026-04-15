# Encoder Selection Guide

Ludwig provides a wide catalog of encoders across all feature types. This guide helps you choose the
right encoder for your task based on your data, accuracy requirements, and computational constraints.

---

## Text Encoders

Text encoders process natural language input. Choose based on your document length, accuracy needs, and
latency budget.

| Encoder | Best For | Speed | Max Tokens | Key Feature |
|---------|----------|-------|------------|-------------|
| `modernbert` | General NLU, classification | Fast | 8192 | SOTA encoder-only, Flash Attention + RoPE |
| `bert` | General NLU baseline | Medium | 512 | Most widely studied, huge ecosystem |
| `roberta` | NLU when more data available | Medium | 512 | Better pretraining than BERT |
| `distilbert` | Speed-sensitive deployment | Fast | 512 | 97% of BERT quality, 6x faster |
| `deberta` | Max benchmark accuracy | Slow | 512 | SOTA on SuperGLUE/SQuAD |
| `longformer` | Long documents | Medium | 4096 | Sliding window + global attention |
| `auto_transformer` | Any HF model | Varies | Varies | Use for GTE, BGE, E5, Nomic, etc. |
| `tf_idf` | Interpretability, tiny data | Very Fast | N/A | Non-neural baseline |

### Text Encoder Decision Tree

```
Text encoder selection:
  Need >512 tokens? --> longformer or modernbert
  Need fastest option? --> distilbert or tf_idf
  Need max accuracy? --> deberta
  Need custom HF model? --> auto_transformer
  Default choice --> modernbert or bert
```

### Tips

- **modernbert** is the recommended default for most NLU tasks. It supports long contexts up to 8192
  tokens and uses Flash Attention for efficient training.
- **auto_transformer** is the escape hatch -- use it to load any model from the Hugging Face Hub
  (e.g., `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-large-en-v1.5`).
- **tf_idf** is useful as a non-neural baseline for comparison, or when you need fully interpretable
  features on very small datasets.
- For classification tasks, set `trainable: false` to use the pretrained encoder as a frozen feature
  extractor, which is faster and avoids overfitting on small data.

---

## Image Encoders

Image encoders process visual input. Pretrained encoders generally outperform training from scratch
unless you have a very specialized domain.

| Encoder | Best For | Pretrained | Key Feature |
|---------|----------|------------|-------------|
| `resnet` | General classification baseline | ImageNet | Well-understood, fast |
| `efficientnet` | Mobile/efficiency | ImageNet | Best accuracy/FLOP trade-off |
| `vit` | High accuracy | ImageNet | Vision Transformer |
| `swin_transformer` | Hierarchical features | ImageNet | Shifted windows, multi-scale |
| `clip` | Zero-shot, multimodal | CLIP data | Text-aligned visual features |
| `dinov2` | Frozen feature extraction | Self-supervised | No labels needed for pretraining |
| `siglip` | Scaled multimodal | SigLIP data | Better scaling than CLIP |
| `convnextv2` | Pure-CNN, high accuracy | FCMAE | GRN + masked autoencoder pretraining |
| `timm` | Any TIMM model | Various | Access to 700+ architectures |
| `stacked_cnn` | Small data, custom arch | None | Fully configurable CNN stack |

### Image Encoder Decision Tree

```
Image encoder selection:
  Need multimodal (image + text)? --> clip or siglip
  Need frozen features, no labels? --> dinov2
  Need max accuracy? --> vit or swin_transformer
  Need efficiency / mobile? --> efficientnet
  Need pure-CNN, no attention? --> convnextv2
  Need a specific TIMM arch? --> timm
  Training from scratch? --> stacked_cnn or resnet
  Default choice --> resnet or vit
```

### Tips

- **timm** gives access to over 700 architectures from the
  [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) library. Use it when you
  need a specific architecture not listed above (e.g., ConvNeXt, EfficientNetV2, MaxViT).
- **clip** and **siglip** produce embeddings aligned with text, making them good choices when you
  combine image and text features in a multimodal model.
- **stacked_cnn** trains from scratch with no pretrained weights -- use it only when your images are
  very different from natural images (e.g., medical scans, spectrograms) and you have enough data.

---

## Audio Encoders

Audio encoders process sound input, either as raw waveforms or spectrograms.

| Encoder | Best For | Input | Key Feature |
|---------|----------|-------|-------------|
| `wav2vec2` | Speech tasks | Raw waveform | Self-supervised, English-first |
| `whisper` | Multilingual, noisy audio | Mel spectrogram | 680K hours training data |
| `hubert` | Speaker/emotion tasks | Raw waveform | Masked prediction pretraining |
| `parallel_cnn` | Simple classification | Spectrogram | Fast, no pretrained weights needed |

### Audio Encoder Decision Tree

```
Audio encoder selection:
  Multilingual or noisy audio? --> whisper
  Speaker ID or emotion? --> hubert
  English speech tasks? --> wav2vec2
  Simple audio classification? --> parallel_cnn
  Default choice --> whisper
```

### Tips

- **whisper** is the most robust choice for general audio tasks. It was trained on 680,000 hours of
  multilingual data and handles noise well.
- **wav2vec2** and **hubert** work directly on raw waveforms -- no spectrogram preprocessing needed.
- **parallel_cnn** is a lightweight option that trains from scratch. Use it when pretrained speech
  models are overkill (e.g., simple environmental sound classification).

---

## Sequence Encoders

Sequence encoders process ordered sequences of tokens (used by sequence, text, and time series
features). Choose based on sequence length and quality requirements.

| Encoder | Best For | Complexity | Key Feature |
|---------|----------|------------|-------------|
| `stacked_transformer` | Medium sequences | O(n^2) | Full attention, best quality |
| `mamba` | Long sequences | O(n) | Linear-time SSM |
| `rnn` / `stacked_rnn` | Short sequences, low memory | O(n) | Sequential processing |
| `parallel_cnn` | Local patterns | O(n) | Parallel convolutions, fast |
| `stacked_cnn` | Hierarchical local patterns | O(n) | Deep convolutional stack |

### Sequence Encoder Decision Tree

```
Sequence encoder selection:
  Very long sequences (>1000 tokens)? --> mamba
  Need best quality, moderate length? --> stacked_transformer
  Need speed, local patterns only? --> parallel_cnn or stacked_cnn
  Low memory budget? --> rnn
  Default choice --> stacked_transformer
```

### Tips

- **mamba** uses a State Space Model (SSM) architecture that scales linearly with sequence length,
  making it ideal for very long sequences where transformer attention would be too expensive.
- **parallel_cnn** applies multiple convolution filters in parallel and is the fastest option. It works
  well when the signal is local (e.g., n-gram patterns).
- **stacked_rnn** supports LSTM and GRU cells. It is memory-efficient but slow to train due to
  sequential processing.

---

## Number Encoders

Number encoders transform scalar numerical inputs into dense representations for the combiner.

| Encoder | Best For | Key Feature |
|---------|----------|-------------|
| `passthrough` | Pre-normalized data | No transformation |
| `dense` | Simple learned projection | Single FC layer |
| `ple` | Tabular SOTA | Piecewise linear, quantile bins |
| `periodic` | Smooth functions | Learned sinusoidal features |
| `bins` | Simple discretization | Bin + embedding, fast |

### Number Encoder Decision Tree

```
Number encoder selection:
  Data already normalized? --> passthrough
  Tabular competition / max accuracy? --> ple
  Periodic or cyclical patterns? --> periodic
  Simple baseline? --> dense
  Default choice --> dense or passthrough
```

### Tips

- **ple** (Piecewise Linear Encoding) discretizes numbers into quantile-based bins with learned linear
  interpolation. It is the current state-of-the-art for tabular data.
- **periodic** learns sinusoidal representations and works well when the underlying function is smooth.
- **passthrough** simply forwards the raw number -- use it when your data is already normalized and you
  want the combiner to handle feature interactions directly.

---

## Category Encoders

Category encoders transform categorical variables into dense or sparse representations.

| Encoder | Best For | Key Feature |
|---------|----------|-------------|
| `dense` | Default, medium cardinality | Learned embedding |
| `sparse` | Large vocab, few active | Sparse embedding lookup |
| `onehot` | Small vocab, tree models | One-hot vector |
| `target` | High cardinality | Mean target encoding |
| `hash` | Extreme cardinality, streaming | Feature hashing, fixed memory |

### Category Encoder Decision Tree

```
Category encoder selection:
  Very high cardinality (>10K)? --> hash or target
  Small vocab (<20 categories)? --> onehot or dense
  Need fixed memory footprint? --> hash
  Default choice --> dense
```

### Tips

- **dense** is the default and works well for most cases. It learns an embedding vector for each
  category.
- **hash** uses feature hashing to map categories to a fixed-size vector. It handles unseen categories
  gracefully and uses constant memory regardless of vocabulary size.
- **target** encoding replaces each category with the mean of the target variable. It is effective for
  high-cardinality features but requires care to avoid target leakage (Ludwig handles this internally
  with cross-fitting).

---

## Date Encoders

Date encoders extract temporal features from date/datetime inputs.

| Encoder | Best For | Key Feature |
|---------|----------|-------------|
| `DateEmbed` | General date features | Embeds each date component (year, month, day, etc.) separately |
| `DateWave` | Cyclical patterns | Uses sinusoidal encoding for cyclical components (month, day of week) |

- **DateEmbed** is the default. It creates separate embeddings for year, month, day, weekday, hour, etc.
  and concatenates them.
- **DateWave** uses sine/cosine transformations for cyclical date components, which helps the model
  understand that December and January are adjacent months.

---

## H3 Encoders

H3 encoders process [Uber H3](https://h3geo.org/) geospatial hex indices.

| Encoder | Best For | Key Feature |
|---------|----------|-------------|
| `H3Embed` | Default geospatial encoding | Embeds H3 components (mode, edge, resolution, cells) separately |
| `H3WeightedSum` | Aggregated representation | Weighted sum of cell embeddings |
| `H3RNN` | Sequential cell patterns | Processes H3 cell hierarchy with an RNN |

- **H3Embed** is the simplest and most common choice. It embeds each component of the H3 index
  independently.
- **H3WeightedSum** learns weights over cell embeddings, useful when spatial aggregation matters.
- **H3RNN** treats the hierarchical cell resolution levels as a sequence, capturing multi-scale spatial
  patterns.

---

## General Recommendations

1. **Start with defaults.** Ludwig's default encoders are reasonable for most tasks. Only change them
   when you have a specific reason.

2. **Use pretrained encoders when possible.** For text, image, and audio features, pretrained encoders
   almost always outperform training from scratch, especially on smaller datasets.

3. **Match encoder to data size.** Larger encoders (e.g., deberta, swin_transformer) need more data to
   fine-tune effectively. On small datasets, use smaller encoders or freeze pretrained weights with
   `trainable: false`.

4. **Consider latency.** For production deployment, encoder inference time matters. Distilled models
   (distilbert, efficientnet) offer good accuracy-speed trade-offs.

5. **Use `auto_transformer` / `timm` as escape hatches.** If none of the named encoders fit your needs,
   these meta-encoders let you load any compatible model from the Hugging Face or TIMM ecosystem.
