The Ludwig Dataset Zoo provides datasets that can be directly plugged into a Ludwig model.

The simplest way to use a dataset is to reference it as a URI when specifying the training set:

```bash
ludwig train --dataset ludwig://reuters ...
```

Any Ludwig dataset can be specified as a URI of the form `ludwig://<dataset>`.

Datasets can also be programatically imported and loaded into a Pandas DataFrame using the `.load()` method:

```python
from ludwig.datasets import reuters

# Loads into single dataframe with a 'split' column:
dataset_df = reuters.load()

# Loads into split dataframes:
train_df, test_df, _ = reuters.load(split=True)
```

The `ludwig.datasets` API also provides functions to list, describe, and get datasets.  For example:

```python
import ludwig.datasets

# Gets a list of all available dataset names.
dataset_names = ludwig.datasets.list_datasets()

# Prints the description of the titanic dataset.
print(ludwig.datasets.describe_dataset("titanic"))

titanic = ludwig.datasets.get_dataset("titanic")

# Loads into single dataframe with a 'split' column:
dataset_df = titanic.load()

# Loads into split dataframes:
train_df, test_df, _ = titanic.load(split=True)
```

## Kaggle Datasets

Some datasets are hosted on [Kaggle](https://www.kaggle.com) and require a kaggle account. To use these, you'll need to
[set up Kaggle credentials](https://www.kaggle.com/docs/api) in your environment. If the dataset is part of a Kaggle
competition, you'll need to accept the terms on the competition page.

To check programmatically, datasets have an `.is_kaggle_dataset` property.

## Downloading, Processing, and Exporting

Datasets are first downloaded into `LUDWIG_CACHE`, which may be set as an environment variable and defaults to
`$HOME/.ludwig_cache`.

Datasets are automatically loaded, processed, and re-saved as parquet files in the cache.

To export the processed dataset, including any files it depends on, use the `.export(output_directory)` method. This
is recommended if the dataset contains media files like images or audio files. File paths are relative to the working
directory of the training process.

```python
from ludwig.datasets import twitter_bots

# Exports twitter bots dataset and image files to the current working directory.
twitter_bots.export(".")
```

## End-to-end Example

Here's an end-to-end example of training a model using the MNIST dataset:

```python
from ludwig.api import LudwigModel
from ludwig.datasets import mnist

# Initializes a Ludwig model
config = {
    "input_features": [{"name": "image_path", "type": "image"}],
    "output_features": [{"name": "label", "type": "category"}],
}
model = LudwigModel(config)

# Loads and splits MNIST dataset
training_set, test_set, _ = mnist.load(split=True)

# Exports the mnist image files to the current working directory.
mnist.export(".")

# Runs model training
results = model.train(training_set=training_set, test_set=test_set, model_name="mnist_model")
train_stats = results.train_stats
```

## Dataset Splits

All datasets in the dataset zoo are provided with a default train/validation/test split. When loading with
`split=False`, the default split will be returned (and is guaranteed to be the same every time). With `split=True`,
Ludwig will randomly re-split the dataset.

!!! note
    Some benchmark or contest datasets are released with held-out test set labels. In other words, the train and
    validation splits have labels, but the test set does not. Most Kaggle contest datasets have this unlabeled test set.

Splits:

- **train**: Data to train on. Required, must have labels.
- **validation**: Subset of dataset to evaluate while training. Optional, must have labels.
- **test**: Held out from model development, used for later testing. Optional, may not be labeled.


## Zoo Datasets

Ludwig ships with **590 datasets** spanning 15 task categories.
Each dataset can be loaded with `ludwig datasets download <name>` or `from ludwig.datasets import <name>`.

### Text Classification (286 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `aegis_safety` | HuggingFace | NVIDIA Aegis 2.0: AI content safety classification. 30K train examples. |
| `ag_news_hf` | HuggingFace | AG News 4-class topic classification (HF version) |
| `agnews` | Download | News articles categorized as "World", "Sports", "Business", and "Science". |
| `amazon_massive_intent` | HuggingFace | Amazon MASSIVE multilingual intent classification (60 intents, all 51 languages combined). 587K trai |
| `amazon_massive_scenario` | HuggingFace | Amazon MASSIVE multilingual scenario classification (18 scenarios, 60 languages). 587K train. |
| `amazon_polarity` | HuggingFace | Amazon product review polarity; positive/negative |
| `amazon_reviews` | Download | The Amazon Reviews dataset |
| `amazon_science_massive` | HuggingFace | Amazon MASSIVE multilingual intent classification dataset (51 languages, 60 intents). |
| `anli` | HuggingFace | Adversarial NLI; iteratively adversarial NLI benchmark |
| `aqua_rat` | HuggingFace | AQuA-RAT: algebraic word problems 5-way MC with rationales. 97K train examples. |
| `arc_challenge` | HuggingFace | AI2 ARC-Challenge harder science QA. 4-way multiple choice. |
| `arc_easy` | HuggingFace | AI2 ARC-Easy science QA. 4-way multiple choice. A/B/C/D. |
| `banking77` | HuggingFace | Banking77; 77-class banking customer intent classification |
| `banking77_legacy` | HuggingFace | Banking77 (legacy) 77-class banking intent classification. 10K train examples. |
| `belebele` | HuggingFace | Belebele multilingual reading comprehension; multiple-choice |
| `belebele_fr` | HuggingFace | Belebele French; multilingual reading comprehension |
| `bitext_customer_intent` | HuggingFace | Bitext customer support intent detection (26K train, 27 intents) |
| `bitext_customer_support` | HuggingFace | Bitext customer support: 27 intents, 10 categories. 26K train examples. |
| `boolq` | HuggingFace | Boolean Questions; reading comprehension yes/no questions from Google |
| `brazilian_toxic_tweets` | HuggingFace | Brazilian toxic tweets binary classification. 8K train examples. |
| `ccnews` | HuggingFace | CC-News: multilingual web news articles from Common Crawl. 1.9M train examples. |
| `climate_fever` | HuggingFace | ClimateFEVER: climate change claim verification dataset. |
| `climate_sentiment` | HuggingFace | ClimateBERT climate-related text sentiment analysis |
| `clinc_oos` | HuggingFace | CLINC OOS: 150-class intent classification with out-of-scope detection. 10K train examples. |
| `clue_afqmc` | HuggingFace | CLUE AFQMC: Chinese short text similarity for financial domain. 34K train examples. |
| `code_contests` | HuggingFace | DeepMind code contests: competitive programming problems with difficulty classification. 3.7K train  |
| `coig_cqia` | HuggingFace | COIG-CQIA: Chinese instruction dataset with domain/task type. 1111 examples. |
| `cola` | HuggingFace | Corpus of Linguistic Acceptability; grammatically acceptable or not |
| `commitment_bank` | HuggingFace | CommitmentBank; textual entailment with 3-way classification |
| `commonsense_qa` | HuggingFace | CommonsenseQA 5-way multiple choice. Question → A/B/C/D/E. |
| `copa` | HuggingFace | Choice Of Plausible Alternatives; causal commonsense reasoning |
| `customer_reviews` | HuggingFace | Customer Reviews; product review sentiment binary classification |
| `dair_emotion` | HuggingFace | DAIR.AI Emotion: 6-class emotion classification from English Twitter messages. |
| `data_scientist_salary` | Download | The training data and test data comprise of 19802 samples and of 6601 samples each from the |
| `databench_qa` | HuggingFace | DataBench: data analysis question-answer type classification. 1830 train examples. |
| `databricks_dolly_15k` | HuggingFace | Databricks Dolly 15K instruction-response pairs with 8 task categories (QA, summarization, etc.). |
| `dbpedia` | Download | The DBPedia Ontology dataset. |
| `dbpedia_14` | HuggingFace | DBpedia 14 ontology text classification; 14 categories |
| `dolly_15k` | HuggingFace | Databricks Dolly 15K: 15K instruction-response pairs with task category labels. |
| `emotion` | HuggingFace | Twitter emotion classification; 6 classes: joy, sadness, anger, fear, surprise, love |
| `enron_spam` | HuggingFace | Enron Spam; email spam/ham classification |
| `fake_news_detection` | HuggingFace | Fake news detection; real vs fake news articles |
| `farstail_nli` | HuggingFace | FarsTail: Persian NLI (entailment/neutral/contradiction). 1K test examples. |
| `fever` | Download | FEVER: a Large-scale Dataset for Fact Extraction and VERification |
| `fever_gold` | HuggingFace | FEVER fact verification gold evidence. Claim → SUPPORTS/REFUTES/NOT_ENOUGH_INFO. |
| `financial_phrasebank` | HuggingFace | Financial classification: sentiment analysis of financial text. |
| `flores_101` | HuggingFace | FLoRes-101 English evaluation sentences. Topic classification: predict |
| `goemotions` | Download | GoEmotions: A Dataset for Fine-Grained Emotion Classification. |
| `google_quest_qa` | Download | Google QUEST Q&A Labeling |
| `hate_speech18` | HuggingFace | Hate Speech 18 (SetFit): binary hate speech detection on online forum data. |
| `hatespeech_offensive` | HuggingFace | Hate Speech and Offensive Language: 25K tweets with 3-class labels |
| `hellaswag` | HuggingFace | HellaSwag commonsense NLI. Activity + context → correct ending. A/B/C/D. |
| `imdb_sentiment` | HuggingFace | IMDB movie review sentiment; positive/negative binary classification |
| `indic_glue` | HuggingFace | IndicGLUE Telugu sentiment classification (actsa-sc). 4328 train examples. |
| `klue_topic` | HuggingFace | KLUE YNAT; Korean news topic classification; 7 categories |
| `kmmlu` | HuggingFace | KMMLU: Korean Massive Multitask Language Understanding, 4-way MC. 45 train examples per subset. |
| `m3cot` | HuggingFace | M3CoT: multimodal multi-domain QA category classification. 7K examples. |
| `m_mmlu` | HuggingFace | Multilingual MMLU (Arabic): 4-way multiple choice academic questions. 274 train examples. |
| `medmcqa` | HuggingFace | Medical entrance exam QA; 4-choice medical questions |
| `melbourne_airbnb` | Download | Melbourne Airbnb Open Data |
| `mmlu` | HuggingFace | MMLU massive multitask benchmark. 57 tasks, 4-way multiple choice. |
| `mmlu_lighteval` | HuggingFace | MMLU: Massive Multitask Language Understanding, 57 academic subjects, 4-way MC. 99K auxiliary train. |
| `mmlu_pro` | HuggingFace | MMLU-Pro harder multitask benchmark. 10-way multiple choice. |
| `mnli` | HuggingFace | Multi-Genre Natural Language Inference; premise + hypothesis -> entailment/neutral/contradiction |
| `mrpc` | HuggingFace | Microsoft Research Paraphrase Corpus; paraphrase detection |
| `mteb_amazon_reviews_class_de` | HuggingFace | MTEB Amazon Reviews Classification (German): 5-class star rating classification. |
| `mteb_amazon_reviews_class_en` | HuggingFace | MTEB Amazon Reviews Classification (English): 5-class star rating classification. |
| `mteb_amazon_reviews_class_es` | HuggingFace | MTEB Amazon Reviews Classification (Spanish): 5-class star rating classification. |
| `mteb_amazon_reviews_class_fr` | HuggingFace | MTEB Amazon Reviews Classification (French): 5-class star rating classification. |
| `mteb_amazon_reviews_class_ja` | HuggingFace | MTEB Amazon Reviews Classification (Japanese): 5-class star rating classification. |
| `mteb_amazon_reviews_class_zh` | HuggingFace | MTEB Amazon Reviews Classification (Chinese): 5-class star rating classification. |
| `mteb_cyrillic_turkic` | HuggingFace | MTEB Cyrillic Turkic Language Classification: language identification for Cyrillic-script Turkic lan |
| `mteb_emotion` | HuggingFace | MTEB emotion task; 6-class emotion from tweets |
| `mteb_financial_phrasebank` | HuggingFace | MTEB Financial Phrasebank Classification: financial news sentiment classification. |
| `mteb_frenk_en` | HuggingFace | MTEB Frenk English Classification: hate speech detection in English. |
| `mteb_frenk_hr` | HuggingFace | MTEB Frenk Croatian Classification: hate speech detection in Croatian. |
| `mteb_frenk_sl` | HuggingFace | MTEB Frenk Slovenian Classification: hate speech detection in Slovenian. |
| `mteb_georeview` | HuggingFace | MTEB Georeview Classification: Russian-language geographic review sentiment classification. |
| `mteb_greek_legal` | HuggingFace | MTEB Greek Legal Code Classification: Greek legal code topic classification. |
| `mteb_ita_casehold` | HuggingFace | MTEB ItaCasehold Classification: Italian legal case holding classification. |
| `mteb_jd_review` | HuggingFace | MTEB JDReview: Chinese product review sentiment classification from JD.com. |
| `mteb_kor_sarcasm` | HuggingFace | MTEB Korean Sarcasm Classification: sarcasm detection in Korean social media. |
| `mteb_language_class` | HuggingFace | MTEB Language Classification: language identification from short text. |
| `mteb_massive_intent_af` | HuggingFace | MTEB MASSIVE Intent Classification (Afrikaans): task-oriented dialog intent prediction. |
| `mteb_massive_intent_am` | HuggingFace | MTEB MASSIVE Intent Classification (Amharic): task-oriented dialog intent prediction. |
| `mteb_massive_intent_ar` | HuggingFace | MTEB MASSIVE Intent (Arabic): 60-class intent classification from voice assistant queries. |
| `mteb_massive_intent_az` | HuggingFace | MTEB MASSIVE Intent Classification (Azerbaijani): task-oriented dialog intent prediction. |
| `mteb_massive_intent_bn` | HuggingFace | MTEB MASSIVE Intent Classification (Bengali): task-oriented dialog intent prediction. |
| `mteb_massive_intent_cy` | HuggingFace | MTEB MASSIVE Intent Classification (Welsh): task-oriented dialog intent prediction. |
| `mteb_massive_intent_da` | HuggingFace | MTEB MASSIVE Intent Classification (Danish): task-oriented dialog intent prediction. |
| `mteb_massive_intent_de` | HuggingFace | MTEB MASSIVE Intent (German): 60-class intent classification from voice assistant queries. |
| `mteb_massive_intent_el` | HuggingFace | MTEB MASSIVE Intent Classification (Greek): task-oriented dialog intent prediction. |
| `mteb_massive_intent_en` | HuggingFace | MTEB MASSIVE Intent (English): 60-class intent classification from voice assistant queries. |
| `mteb_massive_intent_es` | HuggingFace | MTEB MASSIVE Intent (Spanish): 60-class intent classification from voice assistant queries. |
| `mteb_massive_intent_fa` | HuggingFace | MTEB MASSIVE Intent Classification (Farsi): task-oriented dialog intent prediction. |
| `mteb_massive_intent_fi` | HuggingFace | MTEB MASSIVE Intent Classification (Finnish): task-oriented dialog intent prediction. |
| `mteb_massive_intent_fr` | HuggingFace | MTEB MASSIVE Intent (French): 60-class intent classification from voice assistant queries. |
| `mteb_massive_intent_he` | HuggingFace | MTEB MASSIVE Intent Classification (Hebrew): task-oriented dialog intent prediction. |
| `mteb_massive_intent_hi` | HuggingFace | MTEB MASSIVE Intent Classification (Hindi): task-oriented dialog intent prediction. |
| `mteb_massive_intent_hu` | HuggingFace | MTEB MASSIVE Intent Classification (Hungarian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_hy` | HuggingFace | MTEB MASSIVE Intent Classification (Armenian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_id` | HuggingFace | MTEB MASSIVE Intent Classification (Indonesian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_is` | HuggingFace | MTEB MASSIVE Intent Classification (Icelandic): task-oriented dialog intent prediction. |
| `mteb_massive_intent_it` | HuggingFace | MTEB MASSIVE Intent Classification (Italian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_ja` | HuggingFace | MTEB MASSIVE Intent Classification (Japanese): task-oriented dialog intent prediction. |
| `mteb_massive_intent_jv` | HuggingFace | MTEB MASSIVE Intent Classification (Javanese): task-oriented dialog intent prediction. |
| `mteb_massive_intent_ka` | HuggingFace | MTEB MASSIVE Intent Classification (Georgian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_km` | HuggingFace | MTEB MASSIVE Intent Classification (Khmer): task-oriented dialog intent prediction. |
| `mteb_massive_intent_kn` | HuggingFace | MTEB MASSIVE Intent Classification (Kannada): task-oriented dialog intent prediction. |
| `mteb_massive_intent_ko` | HuggingFace | MTEB MASSIVE Intent Classification (Korean): task-oriented dialog intent prediction. |
| `mteb_massive_intent_lv` | HuggingFace | MTEB MASSIVE Intent Classification (Latvian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_ml` | HuggingFace | MTEB MASSIVE Intent Classification (Malayalam): task-oriented dialog intent prediction. |
| `mteb_massive_intent_mn` | HuggingFace | MTEB MASSIVE Intent Classification (Mongolian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_ms` | HuggingFace | MTEB MASSIVE Intent Classification (Malay): task-oriented dialog intent prediction. |
| `mteb_massive_intent_my` | HuggingFace | MTEB MASSIVE Intent Classification (Burmese): task-oriented dialog intent prediction. |
| `mteb_massive_intent_nb` | HuggingFace | MTEB MASSIVE Intent Classification (Norwegian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_nl` | HuggingFace | MTEB MASSIVE Intent Classification (Dutch): task-oriented dialog intent prediction. |
| `mteb_massive_intent_pl` | HuggingFace | MTEB MASSIVE Intent Classification (Polish): task-oriented dialog intent prediction. |
| `mteb_massive_intent_pt` | HuggingFace | MTEB MASSIVE Intent Classification (Portuguese): task-oriented dialog intent prediction. |
| `mteb_massive_intent_ro` | HuggingFace | MTEB MASSIVE Intent Classification (Romanian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_ru` | HuggingFace | MTEB MASSIVE Intent Classification (Russian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_sl` | HuggingFace | MTEB MASSIVE Intent Classification (Slovenian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_sq` | HuggingFace | MTEB MASSIVE Intent Classification (Albanian): task-oriented dialog intent prediction. |
| `mteb_massive_intent_sv` | HuggingFace | MTEB MASSIVE Intent Classification (Swedish): task-oriented dialog intent prediction. |
| `mteb_massive_intent_sw` | HuggingFace | MTEB MASSIVE Intent Classification (Swahili): task-oriented dialog intent prediction. |
| `mteb_massive_intent_ta` | HuggingFace | MTEB MASSIVE Intent Classification (Tamil): task-oriented dialog intent prediction. |
| `mteb_massive_intent_te` | HuggingFace | MTEB MASSIVE Intent Classification (Telugu): task-oriented dialog intent prediction. |
| `mteb_massive_intent_th` | HuggingFace | MTEB MASSIVE Intent Classification (Thai): task-oriented dialog intent prediction. |
| `mteb_massive_intent_tl` | HuggingFace | MTEB MASSIVE Intent Classification (Tagalog): task-oriented dialog intent prediction. |
| `mteb_massive_intent_tr` | HuggingFace | MTEB MASSIVE Intent Classification (Turkish): task-oriented dialog intent prediction. |
| `mteb_massive_intent_ur` | HuggingFace | MTEB MASSIVE Intent Classification (Urdu): task-oriented dialog intent prediction. |
| `mteb_massive_intent_vi` | HuggingFace | MTEB MASSIVE Intent Classification (Vietnamese): task-oriented dialog intent prediction. |
| `mteb_massive_intent_zh_cn` | HuggingFace | MTEB MASSIVE Intent Classification (Chinese (Simplified)): task-oriented dialog intent prediction. |
| `mteb_massive_intent_zh_tw` | HuggingFace | MTEB MASSIVE Intent Classification (Chinese (Traditional)): task-oriented dialog intent prediction. |
| `mteb_massive_scenario_af` | HuggingFace | MTEB MASSIVE Scenario Classification (Afrikaans): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_am` | HuggingFace | MTEB MASSIVE Scenario Classification (Amharic): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_ar` | HuggingFace | MTEB MASSIVE Scenario Classification (Arabic): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_az` | HuggingFace | MTEB MASSIVE Scenario Classification (Azerbaijani): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_bn` | HuggingFace | MTEB MASSIVE Scenario Classification (Bengali): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_cy` | HuggingFace | MTEB MASSIVE Scenario Classification (Welsh): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_da` | HuggingFace | MTEB MASSIVE Scenario Classification (Danish): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_de` | HuggingFace | MTEB MASSIVE Scenario (German): 18-class scenario classification from voice assistant queries. |
| `mteb_massive_scenario_el` | HuggingFace | MTEB MASSIVE Scenario Classification (Greek): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_en` | HuggingFace | MTEB MASSIVE Scenario (English): 18-class scenario classification from voice assistant queries. |
| `mteb_massive_scenario_es` | HuggingFace | MTEB MASSIVE Scenario (Spanish): 18-class scenario classification from voice assistant queries. |
| `mteb_massive_scenario_fa` | HuggingFace | MTEB MASSIVE Scenario Classification (Farsi): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_fi` | HuggingFace | MTEB MASSIVE Scenario Classification (Finnish): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_fr` | HuggingFace | MTEB MASSIVE Scenario (French): 18-class scenario classification from voice assistant queries. |
| `mteb_massive_scenario_he` | HuggingFace | MTEB MASSIVE Scenario Classification (Hebrew): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_hi` | HuggingFace | MTEB MASSIVE Scenario Classification (Hindi): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_hu` | HuggingFace | MTEB MASSIVE Scenario Classification (Hungarian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_hy` | HuggingFace | MTEB MASSIVE Scenario Classification (Armenian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_id` | HuggingFace | MTEB MASSIVE Scenario Classification (Indonesian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_is` | HuggingFace | MTEB MASSIVE Scenario Classification (Icelandic): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_it` | HuggingFace | MTEB MASSIVE Scenario Classification (Italian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_ja` | HuggingFace | MTEB MASSIVE Scenario Classification (Japanese): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_jv` | HuggingFace | MTEB MASSIVE Scenario Classification (Javanese): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_ka` | HuggingFace | MTEB MASSIVE Scenario Classification (Georgian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_km` | HuggingFace | MTEB MASSIVE Scenario Classification (Khmer): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_kn` | HuggingFace | MTEB MASSIVE Scenario Classification (Kannada): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_ko` | HuggingFace | MTEB MASSIVE Scenario Classification (Korean): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_lv` | HuggingFace | MTEB MASSIVE Scenario Classification (Latvian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_ml` | HuggingFace | MTEB MASSIVE Scenario Classification (Malayalam): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_mn` | HuggingFace | MTEB MASSIVE Scenario Classification (Mongolian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_ms` | HuggingFace | MTEB MASSIVE Scenario Classification (Malay): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_my` | HuggingFace | MTEB MASSIVE Scenario Classification (Burmese): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_nb` | HuggingFace | MTEB MASSIVE Scenario Classification (Norwegian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_nl` | HuggingFace | MTEB MASSIVE Scenario Classification (Dutch): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_pl` | HuggingFace | MTEB MASSIVE Scenario Classification (Polish): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_pt` | HuggingFace | MTEB MASSIVE Scenario Classification (Portuguese): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_ro` | HuggingFace | MTEB MASSIVE Scenario Classification (Romanian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_ru` | HuggingFace | MTEB MASSIVE Scenario Classification (Russian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_sl` | HuggingFace | MTEB MASSIVE Scenario Classification (Slovenian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_sq` | HuggingFace | MTEB MASSIVE Scenario Classification (Albanian): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_sv` | HuggingFace | MTEB MASSIVE Scenario Classification (Swedish): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_sw` | HuggingFace | MTEB MASSIVE Scenario Classification (Swahili): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_ta` | HuggingFace | MTEB MASSIVE Scenario Classification (Tamil): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_te` | HuggingFace | MTEB MASSIVE Scenario Classification (Telugu): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_th` | HuggingFace | MTEB MASSIVE Scenario Classification (Thai): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_tl` | HuggingFace | MTEB MASSIVE Scenario Classification (Tagalog): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_tr` | HuggingFace | MTEB MASSIVE Scenario Classification (Turkish): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_ur` | HuggingFace | MTEB MASSIVE Scenario Classification (Urdu): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_vi` | HuggingFace | MTEB MASSIVE Scenario Classification (Vietnamese): task-oriented dialog scenario prediction. |
| `mteb_massive_scenario_zh_cn` | HuggingFace | MTEB MASSIVE Scenario (Chinese Simplified): 18-class scenario classification. |
| `mteb_massive_scenario_zh_tw` | HuggingFace | MTEB MASSIVE Scenario Classification (Chinese (Traditional)): task-oriented dialog scenario predicti |
| `mteb_mtop_domain_de` | HuggingFace | MTEB MTOP Domain Classification (German): task-oriented dialog domain prediction. |
| `mteb_mtop_domain_en` | HuggingFace | MTEB MTOP Domain Classification (English): task-oriented dialog domain prediction. |
| `mteb_mtop_domain_es` | HuggingFace | MTEB MTOP Domain Classification (Spanish): task-oriented dialog domain prediction. |
| `mteb_mtop_domain_fr` | HuggingFace | MTEB MTOP Domain Classification (French): task-oriented dialog domain prediction. |
| `mteb_mtop_domain_hi` | HuggingFace | MTEB MTOP Domain Classification (Hindi): task-oriented dialog domain prediction. |
| `mteb_mtop_domain_th` | HuggingFace | MTEB MTOP Domain Classification (Thai): task-oriented dialog domain prediction. |
| `mteb_mtop_intent_de2` | HuggingFace | MTEB MTOP Intent Classification (German): task-oriented dialog intent prediction. |
| `mteb_mtop_intent_en` | HuggingFace | MTEB MTOP Intent Classification (English): task-oriented dialog intent prediction. |
| `mteb_mtop_intent_es2` | HuggingFace | MTEB MTOP Intent Classification (Spanish): task-oriented dialog intent prediction. |
| `mteb_mtop_intent_fr2` | HuggingFace | MTEB MTOP Intent Classification (French): task-oriented dialog intent prediction. |
| `mteb_mtop_intent_hi2` | HuggingFace | MTEB MTOP Intent Classification (Hindi): task-oriented dialog intent prediction. |
| `mteb_mtop_intent_th2` | HuggingFace | MTEB MTOP Intent Classification (Thai): task-oriented dialog intent prediction. |
| `mteb_multilingual_sentiment` | HuggingFace | MTEB Multilingual Sentiment Classification: multilingual product review sentiment. |
| `mteb_naija_senti_hau` | HuggingFace | MTEB NaijaSenti (Hausa): Nigerian language sentiment classification. |
| `mteb_naija_senti_ibo` | HuggingFace | MTEB NaijaSenti (Igbo): Nigerian language sentiment classification. |
| `mteb_naija_senti_pcm` | HuggingFace | MTEB NaijaSenti (Nigerian Pidgin): Nigerian language sentiment classification. |
| `mteb_naija_senti_yor` | HuggingFace | MTEB NaijaSenti (Yoruba): Nigerian language sentiment classification. |
| `mteb_nepali_news` | HuggingFace | MTEB Nepali News Classification: news category classification in Nepali. |
| `mteb_nordic_lang` | HuggingFace | MTEB Nordic Language Classification: language identification for Nordic languages. |
| `mteb_online_shopping` | HuggingFace | MTEB OnlineShopping: Chinese online shopping review sentiment classification. |
| `mteb_poem_sentiment` | HuggingFace | MTEB Poem Sentiment Classification: sentiment classification of poem verses. |
| `mteb_sensitive_topics` | HuggingFace | MTEB Sensitive Topics Classification: sensitive topic detection in text. |
| `mteb_sentiment_hindi` | HuggingFace | MTEB Sentiment Analysis Hindi: sentiment classification of Hindi text. |
| `mteb_swiss_judgement_de` | HuggingFace | MTEB Swiss Judgement Classification (German): Swiss Federal Supreme Court judgement prediction. |
| `mteb_swiss_judgement_fr` | HuggingFace | MTEB Swiss Judgement Classification (French): Swiss Federal Supreme Court judgement prediction. |
| `mteb_swiss_judgement_it` | HuggingFace | MTEB Swiss Judgement Classification (Italian): Swiss Federal Supreme Court judgement prediction. |
| `mteb_tnews` | HuggingFace | MTEB TNews: Chinese news topic classification dataset. |
| `mteb_turkish_product` | HuggingFace | MTEB Turkish Product Sentiment Classification: Turkish product review sentiment. |
| `mteb_tweet_sentiment` | HuggingFace | MTEB Tweet Sentiment Extraction: 3-class tweet sentiment classification. |
| `mteb_tweet_topic` | HuggingFace | MTEB Tweet Topic Single Classification: single-label topic classification of tweets. |
| `mteb_waimai` | HuggingFace | MTEB Waimai: Chinese food delivery review sentiment classification. |
| `mteb_yahoo_answers` | HuggingFace | MTEB Yahoo Answers Topics Classification: topic classification of Yahoo Answers questions. |
| `multi_nli` | HuggingFace | Multi-Genre NLI; 10 diverse genres, 3-way NLI |
| `nemotron_pii` | HuggingFace | Nemotron PII: document classification by domain type. 100K examples. |
| `nemotron_safety` | HuggingFace | Nemotron Safety Guard v3: prompt/response safety classification. 451K examples. |
| `news_category` | HuggingFace | News Category Dataset: 210K Huffington Post headlines with 42 categories. |
| `news_channel` | Download | Online News Popularity Data Set |
| `nli_zh_all` | HuggingFace | NLI-ZH: Chinese Natural Language Inference dataset merging multiple Chinese |
| `no_robots` | HuggingFace | No Robots: 10K high-quality instruction-following conversations with category labels. |
| `numinamath` | HuggingFace | NuminaMath: competition math problems with verified answers. 131K train examples. |
| `oasst1` | HuggingFace | OpenAssistant OASST1: 161K messages from 35K conversation trees; role classification. |
| `ohsumed_7400` | Kaggle | Ohsumed corpus is extracted from MEDLINE database. MEDLINE is designed for multi-label classificatio |
| `ohsumed_cmu` | Download | OHSUMED is a well-known medical abstracts dataset. It contains 348,566 references, |
| `openbookqa` | HuggingFace | OpenBookQA elementary science 4-way multiple choice. |
| `or_bench` | HuggingFace | OR-Bench: over-refusal benchmark prompt classification. 80K examples. |
| `paws` | HuggingFace | Paraphrase Adversaries from Word Scrambling; challenging paraphrase detection |
| `paws_x` | HuggingFace | PAWS-X multilingual paraphrase identification. 49K train examples. |
| `poem_sentiment` | HuggingFace | Poem Sentiment: verse-level sentiment classification from English poetry. |
| `poem_sentiment_hf` | HuggingFace | Poem Sentiment (Google Research): verse-level sentiment from English poetry. |
| `product_sentiment_machine_hack` | Download | We challenge the machinehackers community to develop a machine learning model |
| `pubmed_qa` | HuggingFace | PubMedQA biomedical QA. Context + question → yes/no/maybe. |
| `qasc` | HuggingFace | QASC: 8-way MC QA with science facts. 8134 train examples. |
| `qnli` | HuggingFace | Question-answering NLI; whether context sentence contains answer to question |
| `qqp` | HuggingFace | Quora Question Pairs; whether two questions are semantically equivalent |
| `race` | HuggingFace | RACE: large-scale reading comprehension from Chinese English exams. 87K train examples. 4-way MC. |
| `reuters_cmu` | Download | Reuters-21578 is a well-known newswire dataset containing 21,578 documents. |
| `reuters_r8` | Kaggle | Reuters R8 subset of Reuters 21578 dataset from Kaggle. |
| `reward_bench` | HuggingFace | RewardBench: preference evaluation - predict which model response is chosen/preferred. |
| `rotten_tomatoes` | HuggingFace | Rotten Tomatoes movie review sentiment; positive/negative |
| `rte` | HuggingFace | Recognizing Textual Entailment; entailment vs not-entailment |
| `scienceqa_vqa` | HuggingFace | ScienceQA: science question + optional lecture → multiple choice answer. |
| `scotus_classification` | HuggingFace | LexGLUE SCOTUS; US Supreme Court opinion issue area classification |
| `setfit_ag_news` | HuggingFace | SetFit AG News: 4-class news topic classification (World, Sports, Business, Sci/Tech). |
| `setfit_emotion` | HuggingFace | SetFit Emotion: 6-class emotion classification from English Twitter messages. |
| `setfit_yelp_review` | HuggingFace | SetFit Yelp Review Full: 5-class star rating classification from Yelp reviews. |
| `sib200` | HuggingFace | SIB-200: multilingual 7-class topic classification in 205 languages. 143K train examples. |
| `sms_spam` | HuggingFace | SMS spam detection: ham or spam binary classification. 5574 examples. |
| `snli` | HuggingFace | Stanford NLI; premise/hypothesis pairs -> entailment/neutral/contradiction |
| `spotify_tracks` | HuggingFace | Spotify tracks: predict genre from audio features. 114K examples. |
| `sst2_hf` | HuggingFace | Stanford Sentiment Treebank 2-class (HF canonical version) |
| `sst3` | HuggingFace | Three-class sentiment dataset (negative/neutral/positive). |
| `sst5` | HuggingFace | The SST5 dataset (Stanford Sentiment Treebank, 5-class). |
| `sst5_setfit` | HuggingFace | SST-5 fine-grained sentiment (SetFit version); 5 classes |
| `superglue_rte` | HuggingFace | SuperGLUE version of Recognizing Textual Entailment |
| `synthia` | HuggingFace | SYNTHIA: synthetic instructional examples across 20 academic fields. 1.2M examples. |
| `tweet_eval_emoji` | HuggingFace | TweetEval emoji: predict emoji from tweet text. 20-class classification. |
| `tweet_sentiment_extraction` | HuggingFace | Tweet sentiment extraction; positive/negative/neutral |
| `tweeteval_emotion` | HuggingFace | TweetEval emotion classification; 4 classes |
| `tweeteval_hate` | HuggingFace | TweetEval hate speech detection |
| `tweeteval_irony` | HuggingFace | TweetEval irony detection |
| `tweeteval_offensive` | HuggingFace | TweetEval offensive language detection |
| `tweeteval_sentiment` | HuggingFace | TweetEval sentiment; positive/negative/neutral tweet classification |
| `tweeteval_stance` | HuggingFace | TweetEval stance detection; against/in favor/neutral |
| `twitter_financial_news_topic` | HuggingFace | Twitter financial news 20-class topic classification. 17K train examples. |
| `wic` | HuggingFace | Word-in-Context; word sense disambiguation as binary classification |
| `wiki_qa` | HuggingFace | WikiQA: answer sentence selection (relevant/not-relevant). 20K train examples. |
| `wildchat` | HuggingFace | WildChat: 570K real ChatGPT user interactions; language + toxicity classification. |
| `winograd_schema` | HuggingFace | Winograd Schema Challenge; pronoun coreference resolution |
| `winogrande_hf` | HuggingFace | WinoGrande (allenai): large-scale commonsense reasoning benchmark (273K examples). |
| `wnli` | HuggingFace | Winograd NLI; pronoun reference coreference classification |
| `xnli` | HuggingFace | XNLI cross-lingual NLI. Premise+hypothesis -> entailment/neutral/contradiction. |
| `xnli_de` | HuggingFace | XNLI German: cross-lingual natural language inference, German split. |
| `xnli_en` | HuggingFace | XNLI English: cross-lingual natural language inference, English split. |
| `xnli_es` | HuggingFace | XNLI Spanish: cross-lingual natural language inference, Spanish split. |
| `xnli_fr` | HuggingFace | XNLI French: cross-lingual natural language inference, French split. |
| `xnli_zh` | HuggingFace | XNLI Chinese: cross-lingual natural language inference, Chinese split. |
| `yahoo_answers` | Download | The Yahoo Answers dataset |
| `yahoo_answers_topics` | HuggingFace | Yahoo Answers 10-class topic classification. 1.4M train examples. |
| `yelp_polarity` | HuggingFace | Yelp binary sentiment classification: 1 (negative) or 2 (positive). 560K train examples. |
| `yelp_review_full` | HuggingFace | Yelp review 5-star rating classification |
| `yelp_reviews` | Download | The Yelp Reviews dataset |

### Text Generation / Summarization / Translation (82 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `aeslc` | HuggingFace | AESLC: annotated email subject line corpus for email body → subject summarization. |
| `alpaca` | Download | Stanford Alpaca instruction-tuning dataset (https://github.com/tatsu-lab/stanford_alpaca) for LLM fi |
| `alpaca_cleaned` | HuggingFace | Alpaca Cleaned: 52K instruction-following samples, cleaned version of Stanford Alpaca. |
| `alpaca_gpt4` | HuggingFace | Alpaca GPT-4: 52K instruction-following examples. |
| `alpaca_gpt4_zh` | HuggingFace | Alpaca GPT-4 Chinese: 42K Chinese instruction-following examples. |
| `ambig_qa` | HuggingFace | AmbigQA: open-domain QA with ambiguous questions. Text-in text-out task. |
| `arxiv_abstracts_2021` | HuggingFace | ArXiv abstracts 2021: predict abstract from title. 2M train examples. |
| `arxiv_summarization` | HuggingFace | ArXiv document summarization: predict abstract from full paper text. 203K train examples. |
| `bbh` | HuggingFace | Big-Bench Hard boolean expressions. Input → target. |
| `big_patent` | HuggingFace | BigPatent: patent claim summarization (category A). 1.2M train examples. |
| `bigbench` | HuggingFace | BigBench: abstract narrative understanding task. 2400 train examples. |
| `billsum` | HuggingFace | BillSum; US congressional and California bill summarization |
| `bornholm_bitext` | HuggingFace | Bornholm Bitext Mining: Danish-Bornholmsk low-resource translation. 5785 examples. |
| `cmrc2018` | HuggingFace | CMRC 2018: Chinese machine reading comprehension (SQuAD-style). 10K train examples. |
| `cnn_dailymail` | HuggingFace | CNN/DailyMail news summarization. Article -> highlights. ~300K examples. |
| `cnn_dm_hf` | HuggingFace | CNN/DailyMail (abisee): news article summarization, 3.0.0 version (287K examples). |
| `code_alpaca` | Download | This dataset, created by sahil280114, aims to build and share an instruction-following LLaMA model f |
| `code_search_net` | HuggingFace | CodeSearchNet Python: function code → docstring (text generation). |
| `codex_thinking` | HuggingFace | CodeX 2M: code generation with chain-of-thought reasoning. 2.2M examples. |
| `codexglue_code_to_text` | HuggingFace | CodeXGlue Python code → docstring generation. |
| `consumer_complaints` | Kaggle | The dataset contains different information of complaints that customers have made about a multiple p |
| `dialogsum` | HuggingFace | DialogSum dialogue summarization. Dialogue -> summary. ~13K examples. |
| `drop` | HuggingFace | DROP: Discrete Reasoning Over Paragraphs reading comprehension. 77K train examples. |
| `duorc` | HuggingFace | DuoRC SelfRC: movie plot + question → answer text. |
| `europarl_bg_cs` | HuggingFace | Europarl: European Parliament proceedings Bulgarian-Czech translation. 400K train pairs. |
| `europarl_bg_en` | HuggingFace | Europarl: European Parliament proceedings Bulgarian-English translation. |
| `europarl_cs_en` | HuggingFace | Europarl: European Parliament proceedings Czech-English translation. |
| `europarl_da_en` | HuggingFace | Europarl: European Parliament proceedings Danish-English translation. |
| `europarl_de_en` | HuggingFace | Europarl: European Parliament proceedings German-English translation. |
| `europarl_el_en` | HuggingFace | Europarl: European Parliament proceedings Greek-English translation. |
| `europarl_en_es` | HuggingFace | Europarl: European Parliament proceedings English-Spanish translation. |
| `europarl_en_fr` | HuggingFace | Europarl: European Parliament proceedings English-French translation. |
| `europarl_en_it` | HuggingFace | Europarl: European Parliament proceedings English-Italian translation. |
| `europarl_en_nl` | HuggingFace | Europarl: European Parliament proceedings English-Dutch translation. |
| `europarl_en_pl` | HuggingFace | Europarl: European Parliament proceedings English-Polish translation. |
| `europarl_en_pt` | HuggingFace | Europarl: European Parliament proceedings English-Portuguese translation. |
| `europarl_en_ro` | HuggingFace | Europarl: European Parliament proceedings English-Romanian translation. |
| `europarl_en_sv` | HuggingFace | Europarl: European Parliament proceedings English-Swedish translation. |
| `flashrag_2wikimultihop` | HuggingFace | FlashRAG 2WikiMultiHopQA: multi-hop QA. 15K train examples. |
| `gaiasky_qa` | HuggingFace | Gaiasky astronomy Q&A dataset. 3.8K examples. |
| `govreport_summarization` | HuggingFace | GovReport: long government report summarization. 17K train examples. |
| `gsm8k` | HuggingFace | GSM8K; grade school math word problems with step-by-step solutions |
| `gsm8k_openai` | HuggingFace | GSM8K (OpenAI): grade school math word problems (8.5K problems). |
| `hh_rlhf` | HuggingFace | Anthropic HH-RLHF: 170K human feedback pairs for helpful/harmless RLHF training. |
| `hotpot_qa` | HuggingFace | HotpotQA multi-hop reasoning QA. Question → answer. |
| `kilt_nq` | HuggingFace | KILT NQ: Natural Questions in the KILT knowledge-intensive framework. 87K train examples. |
| `language_identification` | HuggingFace | Language identification; 20 languages from Twitter data |
| `math500` | HuggingFace | MATH-500 test subset; competition math with step-by-step solutions |
| `mathvista` | HuggingFace | MathVista: math reasoning over images. Question → answer. |
| `mbpp` | HuggingFace | MBPP Python problem description → solution code generation. 500 problems. |
| `medical_flashcards` | HuggingFace | Medical flashcards: Q&A for medical topics. 34K examples. |
| `multi30k` | HuggingFace | Multi30k: English-German image caption translation. 29K train pairs. |
| `multiun_ar_en` | HuggingFace | MultiUN Arabic-English United Nations document translation. 9.8M train pairs. |
| `natural_questions` | HuggingFace | Natural Questions: real Google search questions with Wikipedia answers. |
| `natural_questions_hard_negatives` | HuggingFace | Natural Questions hard negatives for retrieval/ranking |
| `naver_news_summary` | HuggingFace | Naver News Korean summarization dataset |
| `opus100_en_es` | HuggingFace | OPUS-100 English-Spanish parallel corpus. ~1M sentence pairs. |
| `opus100_en_fr` | HuggingFace | OPUS-100 English-French parallel corpus. ~1M sentence pairs. |
| `opus_books_en_fr` | HuggingFace | OPUS Books English-French literary translations. |
| `orca_dpo_pairs` | HuggingFace | Intel Orca DPO Pairs: preference dataset for direct preference optimization. |
| `orca_math` | HuggingFace | OrcaMath: 200K math word problems with solutions. |
| `phinc` | HuggingFace | PHINC: Hindi-English parallel corpus. 13K train pairs. |
| `pubmed_summarization` | HuggingFace | PubMed biomedical document summarization. 120K train examples. |
| `python_code_instructions` | HuggingFace | Python code generation from instructions. 18K examples. |
| `samsum` | HuggingFace | SAMSum: dialogue summarization. 14K train examples. |
| `sciq` | HuggingFace | SciQ science QA with support text. Support + question → correct_answer (text). |
| `scitail` | HuggingFace | SciTail; science textual entailment from multiple-choice questions |
| `setimes_bg_bs` | HuggingFace | SETimes: South-East European Times Bulgarian-Bosnian translation. 136K train pairs. |
| `squad` | HuggingFace | SQuAD v1.1 extractive QA. Context + question → answer text. 87K examples. |
| `squad_v2` | HuggingFace | SQuAD v2 extractive QA with unanswerable questions. 130K examples. |
| `tofu` | HuggingFace | TOFU: Fictitious Unlearning. Fictional author QA dataset for LLM unlearning research. 4K train examp |
| `trivia_qa` | HuggingFace | TriviaQA reading comprehension. Question → answer. |
| `truthful_qa` | HuggingFace | TruthfulQA: question → best truthful answer. 817 questions. |
| `vukuzenzele` | HuggingFace | Vukuzenzele: Afrikaans-English sentence pairs. 2.7K train examples. |
| `web_questions` | HuggingFace | WebQuestions: Freebase-grounded factoid QA. 3.8K train examples. |
| `winogrande` | HuggingFace | WinoGrande; large-scale Winograd schema challenge |
| `wmt14_de_en` | HuggingFace | WMT14 German-English news translation. ~4.5M sentence pairs. |
| `wmt16_de_en` | HuggingFace | WMT16 German-English news translation. |
| `wmt19_de_en` | HuggingFace | WMT19 German-English news translation. |
| `wmt_t2t_de_en` | HuggingFace | WMT T2T German-English news translation. 4.6M train sentence pairs. |
| `xsum` | HuggingFace | XSum BBC news summarization. Document -> one-sentence summary. ~200K examples. |
| `xsum_hf` | HuggingFace | XSum (Edinburgh NLP): extreme summarization of BBC news articles (226K examples). |

### Text Regression (91 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `adult_income_hf` | HuggingFace | Adult/Census Income dataset for income >50K classification |
| `ae_price_prediction` | Download | Innerwear Data from Victoria's Secret and Others |
| `allocine` | HuggingFace | AlloCine: French movie review sentiment dataset. Binary positive/negative labels. |
| `amazon_review_polarity` | Download | The Amazon Reviews Polarity dataset |
| `amazon_reviews_2023` | HuggingFace | Amazon Reviews 2023: predict star rating from review title and text. 571M train examples. |
| `app_reviews` | HuggingFace | App Reviews: 288K mobile app reviews with 1-5 star ratings. |
| `beavertails` | HuggingFace | BeaverTails LLM safety: prompt+response → is_safe binary + category. |
| `blimp` | HuggingFace | BLiMP: Benchmark of Linguistic Minimal Pairs. Binary grammaticality |
| `bookprice_prediction` | Download | Here we explore a database of books of different genres, from thousands of authors. |
| `boolq_standalone` | HuggingFace | BoolQ standalone; naturally occurring yes/no questions with passage |
| `california_house_price` | Download | Predict house sale prices based on the house information, such as # of bedrooms, |
| `civil_comments` | HuggingFace | Civil Comments toxicity classification; multi-attribute toxicity labels |
| `code_defect_detection` | HuggingFace | CodeXGLUE: binary defect detection in code functions. 21K train examples. |
| `factcheck` | HuggingFace | FactCheck: multilingual fact-checking question ranking dataset. 2M train examples. |
| `fake_job_postings2` | Download | This dataset contains 18K job descriptions out of which about 800 are fake. |
| `fineweb_edu` | HuggingFace | FineWeb-Edu: 1.3T token high-quality educational web text; 10BT sample subset. |
| `germeval18` | HuggingFace | GermEval 2018: German offensive language detection. Binary and multi-class labels. |
| `google_qa_answer_type_reason_explanation` | Download | Google QUEST Q&A Labeling |
| `google_qa_question_type_reason_explanation` | Download | Google QUEST Q&A Labeling |
| `hc3` | HuggingFace | Human ChatGPT Comparison Corpus (HC3): binary classification of whether an |
| `hc3_chinese` | HuggingFace | HC3-Chinese: Chinese Human ChatGPT Comparison Corpus. Binary classification |
| `helpsteer2` | HuggingFace | HelpSteer2: 21K prompt-response pairs with 5 quality attributes rated 0-4. |
| `imdb` | Kaggle | IMDB dataset having 50K movie reviews for natural language processing or Text analytics. |
| `imdb_genre_prediction` | Download | A data set of 1,000 most popular movies on IMDB in the last 10 years. The data points included are: |
| `imdb_mteb` | HuggingFace | IMDB movie review sentiment binary classification (positive/negative). 24K train examples. |
| `irony` | Download | The Reddit Irony dataset. |
| `jc_penney_products` | Download | JCPenney products |
| `jigsaw_unintended_bias` | Download | A dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias. |
| `jigsaw_unintended_bias100k` | Download | A dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias. |
| `kick_starter_funding` | Download | Funding Successful Projects on Kickstarter |
| `klue_sts` | HuggingFace | KLUE STS Korean sentence similarity. Sentence pair → score (0-5 regression). |
| `lmsys_arena` | HuggingFace | LMSYS Arena: 55K human preference comparisons between LLM responses. |
| `measuring_hate_speech` | HuggingFace | Measuring Hate Speech; continuous hate speech score regression |
| `mercari_price_suggestion` | Download | Predict product price based on details like product category name, brand name, and item condition. |
| `mercari_price_suggestion100K` | Download | Predict product price based on details like product category name, brand name, and item condition. |
| `moral_stories` | HuggingFace | Moral Stories: binary classification of moral vs immoral actions given a norm, |
| `msmarco_passage` | HuggingFace | MS MARCO passage retrieval; query-passage relevance scoring |
| `mteb_amazon_polarity` | HuggingFace | MTEB Amazon Polarity: binary positive/negative sentiment from Amazon reviews (4M reviews). |
| `mteb_biosses` | HuggingFace | MTEB BIOSSES: biomedical sentence similarity benchmark (100 sentence pairs). |
| `mteb_imdb` | HuggingFace | MTEB IMDB: binary movie review sentiment classification. |
| `mteb_sts17_ar` | HuggingFace | MTEB STS17 (Arabic-Arabic): Arabic semantic textual similarity. |
| `mteb_sts17_de` | HuggingFace | MTEB STS17 (German-English): cross-lingual semantic textual similarity. |
| `mteb_sts17_en` | HuggingFace | MTEB STS17 (English-English): semantic textual similarity regression. |
| `mteb_sts17_es` | HuggingFace | MTEB STS17 (Spanish-English): cross-lingual semantic textual similarity. |
| `mteb_sts17_fr` | HuggingFace | MTEB STS17 (French-English): cross-lingual semantic textual similarity. |
| `mteb_sts22_ar` | HuggingFace | MTEB STS22 (Arabic): semantic textual similarity regression. |
| `mteb_sts22_de` | HuggingFace | MTEB STS22 (German): semantic textual similarity regression. |
| `mteb_sts22_de_en` | HuggingFace | MTEB STS22 (German-English cross-lingual): semantic textual similarity. |
| `mteb_sts22_de_fr` | HuggingFace | MTEB STS22 (German-French cross-lingual): semantic textual similarity. |
| `mteb_sts22_en` | HuggingFace | MTEB STS22 (English): semantic textual similarity regression. |
| `mteb_sts22_es` | HuggingFace | MTEB STS22 (Spanish): semantic textual similarity regression. |
| `mteb_sts22_es_en` | HuggingFace | MTEB STS22 (Spanish-English cross-lingual): semantic textual similarity. |
| `mteb_sts22_es_it` | HuggingFace | MTEB STS22 (Spanish-Italian cross-lingual): semantic textual similarity. |
| `mteb_sts22_fr` | HuggingFace | MTEB STS22 (French): semantic textual similarity regression. |
| `mteb_sts22_it` | HuggingFace | MTEB STS22 (Italian): semantic textual similarity regression. |
| `mteb_sts22_pl` | HuggingFace | MTEB STS22 (Polish): semantic textual similarity regression. |
| `mteb_sts22_pl_en` | HuggingFace | MTEB STS22 (Polish-English cross-lingual): semantic textual similarity. |
| `mteb_sts22_ru` | HuggingFace | MTEB STS22 (Russian): semantic textual similarity regression. |
| `mteb_sts22_tr` | HuggingFace | MTEB STS22 (Turkish): semantic textual similarity regression. |
| `mteb_sts22_zh` | HuggingFace | MTEB STS22 (Chinese): semantic textual similarity regression. |
| `mteb_sts22_zh_en` | HuggingFace | MTEB STS22 (Chinese-English cross-lingual): semantic textual similarity. |
| `mteb_stsbenchmark` | HuggingFace | MTEB STSBenchmark: STS Benchmark semantic similarity (8K sentence pairs, scores 0-5). |
| `mteb_toxic_convo` | HuggingFace | MTEB Toxic Conversations 50K: binary toxicity classification. |
| `multirc` | HuggingFace | SuperGLUE MultiRC: paragraph + question + answer → binary (answer correct?). |
| `news_popularity2` | Download | Online News Popularity Data Set |
| `persuasion` | HuggingFace | Anthropic Persuasion: rate how persuasive arguments are for various claims. |
| `sarcastic_headlines` | Kaggle | A dataset to determine if a news headline is sarcastic or serious. |
| `scandisent` | HuggingFace | ScandiSent: Nordic language sentiment classification. 50K train examples. |
| `setfit_amazon_polarity` | HuggingFace | SetFit Amazon Polarity: binary positive/negative sentiment from Amazon reviews. |
| `setfit_mrpc` | HuggingFace | SetFit MRPC: Microsoft Research Paraphrase Corpus, binary paraphrase detection. |
| `setfit_sst2` | HuggingFace | SetFit SST-2: Stanford Sentiment Treebank binary sentiment (SetFit format). |
| `setfit_subj` | HuggingFace | SetFit SUBJ: subjectivity detection (subjective vs objective sentences). |
| `sickr` | HuggingFace | SICK-R: sentences involving compositional knowledge. 9927 test examples. |
| `sst2` | HuggingFace | The SST2 dataset (Stanford Sentiment Treebank, binary). |
| `stackoverflow_posts` | HuggingFace | Stack Overflow posts: predict question score from title and body. 58M train examples. |
| `sts12` | HuggingFace | STS 2012: semantic textual similarity benchmark. 2234 train examples. |
| `sts13` | HuggingFace | STS 2013: semantic textual similarity. 1500 test examples. |
| `sts14` | HuggingFace | STS 2014: semantic textual similarity. 3750 test examples. |
| `sts15` | HuggingFace | STS 2015: semantic textual similarity. 3000 test examples. |
| `sts16` | HuggingFace | STS 2016: semantic textual similarity. 1186 test examples. |
| `sts17` | HuggingFace | STS 2017: cross-lingual STS. 5346 test examples. |
| `sts22` | HuggingFace | STS 2022: cross-lingual semantic textual similarity. 4.6K train examples. |
| `sts_benchmark` | HuggingFace | STS Benchmark; sentence pair semantic similarity scoring |
| `stsb` | HuggingFace | Semantic Textual Similarity Benchmark; similarity score 0-5 |
| `stsb_de` | HuggingFace | STS Benchmark German (machine translated). 5749 train examples. |
| `stsb_sentencetransformers` | HuggingFace | STS Benchmark: semantic textual similarity scoring (0-5 scale). 5.7K train examples. |
| `student_performance` | HuggingFace | Student performance regression; predict final grade from demographics |
| `toxic_chat` | HuggingFace | LMSys ToxicChat: human toxicity annotation for LLM conversations. |
| `wine_reviews` | Download | Wine Reviews |
| `women_clothing_review` | Download | Women's E-Commerce Clothing Reviews |
| `yelp_review_polarity` | Download | The Yelp Polarity dataset |

### Sequence Tagging / NER (12 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `acronym_identification` | HuggingFace | Acronym identification: tokens → B-long/I-long/B-short/I-short/O tags. |
| `audioset_balanced` | HuggingFace | AudioSet balanced: audio event classification with 527 sound classes from YouTube. 18K train example |
| `few_nerd` | HuggingFace | Few-NERD fine-grained NER with 8 coarse and 66 fine entity types. |
| `multinerd` | HuggingFace | MultiNERD multilingual NER. 31 fine-grained entity types. CC BY 4.0. |
| `nq_open` | HuggingFace | Natural Questions Open: open-domain QA with Wikipedia answers. 88K train examples. |
| `pii_masking` | HuggingFace | PII masking: source text → BIO labels for PII tokens. |
| `universal_dependencies` | HuggingFace | Universal Dependencies English EWT: POS tagging with UPOS tags. 12K train sentences. |
| `wikiann` | HuggingFace | WikiANN (PAN-X) Named Entity Recognition — English |
| `wikiann_de` | HuggingFace | WikiANN German NER. IOB2 tags: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC. |
| `wikiann_en` | HuggingFace | WikiANN English NER. IOB2 tags: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC. |
| `wikiann_zh` | HuggingFace | WikiANN Chinese NER. IOB2 tags: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC. |
| `winobias` | HuggingFace | WinoBias coreference gender bias. Sentence → label. |

### Multi-label Classification (4 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `go_emotions` | HuggingFace | GoEmotions: Multi-label Emotion Classification |
| `go_emotions_multiclass` | HuggingFace | GoEmotions multi-label emotion (28 classes). Set output. Same as go_emotions but listed separately. |
| `lex_glue_ecthr` | HuggingFace | LexGLUE ECtHR case text → violated ECHR articles (multi-label). Set output. |
| `lex_glue_eurlex` | HuggingFace | LexGLUE EuroLex EU documents → EuroVoc concept labels (multi-label). Set output. |

### Image Classification (29 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `ai_generated_ecommerce` | HuggingFace | AI-generated vs real e-commerce product images. 6K examples. |
| `beans` | HuggingFace | Beans leaf disease classification; 3 classes: angular leaf spot, bean rust, healthy |
| `cifar10` | HuggingFace | CIFAR-10; 10-class 32x32 image classification |
| `cifar100` | HuggingFace | CIFAR-100; 100-class 32x32 image classification |
| `eurosat` | HuggingFace | EuroSAT; land use/cover classification from satellite imagery; 10 classes |
| `eurosat_rgb` | HuggingFace | EuroSAT RGB: Sentinel-2 satellite image classification (10 land use classes). 16K train examples. |
| `fashion_mnist` | HuggingFace | Fashion MNIST; 10-class clothing item classification |
| `food101` | HuggingFace | Food-101; 101-class food image classification |
| `graid_bdd` | HuggingFace | GrAID-BDD: grounded autonomous driving image QA. 4.6M train examples. |
| `gtsrb` | HuggingFace | GTSRB; German Traffic Sign Recognition Benchmark; 43 classes |
| `handwritten_crossouts` | HuggingFace | Handwritten cross-outs: classify handwriting correction styles. 22K examples. |
| `imagenet_100` | HuggingFace | ImageNet-100: 100-class subset of ImageNet. 126K train examples. |
| `intuitive_physics` | HuggingFace | Intuitive Physics: image-based physical intuition classification. 280K train. |
| `kvasir_vqa` | HuggingFace | Kvasir-VQA: gastrointestinal endoscopy visual QA. 143K train examples. |
| `map_trace` | HuggingFace | MapTrace: map-type image classification (7 categories). 20K examples. |
| `mini_imagenet` | HuggingFace | Mini-ImageNet: 100-class image classification subset of ImageNet. 50K train examples. |
| `mnist` | Download | The MNIST database of handwritten digits, available from this page, |
| `mnist_ylecun` | HuggingFace | MNIST handwritten digit recognition. 60K train examples. |
| `newyorker_caption_contest` | HuggingFace | New Yorker Caption Contest — Multimodal Image+Text Classification |
| `openfake` | HuggingFace | OpenFake: AI-generated vs real image binary classification. |
| `oxford_pets` | HuggingFace | Oxford-IIIT Pet; 37 pet breed classification |
| `path_vqa` | HuggingFace | PathVQA: pathology visual QA (yes/no + open). 20K train examples. |
| `rendered_sst2` | HuggingFace | Rendered SST2; sentiment classification from rendered text images |
| `stanford_cars` | HuggingFace | Stanford Cars; 196-class fine-grained car make/model/year classification |
| `sun397` | HuggingFace | SUN397; scene understanding; 397 scene categories |
| `svhn` | HuggingFace | SVHN; Street View House Numbers digit classification |
| `tiny_imagenet` | HuggingFace | Tiny ImageNet; 200-class 64x64 subset of ImageNet |
| `tobacco_document` | HuggingFace | Tobacco document image classification with OCR text. 2.2K examples. |
| `wikiart` | HuggingFace | WikiArt; artwork style classification across 27 art styles |

### Document Understanding / VQA (8 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `ai2d_diagrams` | HuggingFace | AI2 Diagrams VQA: diagram image + question → answer. |
| `cord_v2` | HuggingFace | CORD v2 receipt understanding: receipt image -> JSON ground truth (text generation). |
| `docvqa` | HuggingFace | DocVQA: document image + question → answer from the document. |
| `invoice_data` | HuggingFace | Invoice document understanding: invoice image -> JSON ground truth (text generation). |
| `merit` | HuggingFace | MERIT: historical document image recognition. 7K train examples. |
| `textvqa` | HuggingFace | TextVQA: image with text + question → answer requiring reading the text. |
| `vqa_rad` | HuggingFace | VQA-RAD; radiology visual question answering |
| `vqav2` | HuggingFace | VQAv2: image + question → free-form answer. |

### Semantic Segmentation (1 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `camseq` | Kaggle | CamSeq01 Cambridge Labeled Objects in Video |

### Audio Classification (8 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `abjad_kids` | HuggingFace | Abjad-Kids: Arabic letter audio classification (28 classes). 40K examples. |
| `emodb` | HuggingFace | EMo-DB; Berlin Database of Emotional Speech; 7 emotion classes |
| `esc50` | Download | ESC-50: Environmental Sound Classification |
| `minds14` | HuggingFace | MINDS-14: multilingual banking intent classification from audio. 8K train examples. |
| `mmsulab` | HuggingFace | MMSuLab: multilingual audio speech laboratory data. 1.9M train examples. |
| `naturelm_audio` | HuggingFace | NatureLM Audio: wildlife/nature audio instruction following. 26M train examples. |
| `speech_massive` | HuggingFace | Speech-MASSIVE: multilingual audio intent classification. 23K train examples. |
| `tadabur` | HuggingFace | Tadabur: Arabic Quran audio with reciter, surah, and ayah classification. 409K examples. |

### Speech Recognition (ASR) (7 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `ami_asr` | HuggingFace | AMI corpus: meeting audio transcription. 108K train examples. |
| `cantonese_asr` | HuggingFace | Cantonese speech recognition dataset |
| `librispeech` | HuggingFace | LibriSpeech; English speech recognition from audiobooks; clean 100h split |
| `mls_german` | HuggingFace | Multilingual LibriSpeech German ASR |
| `peoples_speech` | HuggingFace | People's Speech; 30K-hour diverse English ASR |
| `ravnursson_asr` | HuggingFace | Ravnursson: Faroese audio speech recognition dataset. 65K train examples. |
| `voxpopuli` | HuggingFace | VoxPopuli; European Parliament speech ASR; 23 languages |

### Tabular Classification (12 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `bbcnews` | Download | BBC News Classification from Kaggle. |
| `connect4` | Kaggle | Each row represents the end results of a Connect-4 game. |
| `diabetes_readmission` | HuggingFace | Diabetes hospital readmission prediction |
| `forest_cover` | Download | The Forest Cover Type dataset. |
| `hermes_function_calling` | HuggingFace | Hermes function calling: tool-use conversations with category labels. 1893 train examples. |
| `iris` | Download | Iris Dataset |
| `iris_sklearn` | HuggingFace | Iris dataset: classic 3-class flower classification by petal/sepal measurements. |
| `mushroom_edibility` | Download | This data set includes descriptions of hypothetical samples corresponding |
| `otto_group_product` | Download | The Otto Group Product Classification Challenge |
| `poker_hand` | Download | Each record is an example of a hand consisting of five playing cards |
| `taix_ray` | HuggingFace | TAIX-Ray: thoracic X-ray study - predict patient sex from age and other metadata. 137K train example |
| `walmart_recruiting` | Download | Walmart Recruiting: Trip Type Classification |

### Tabular Binary Classification (24 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `adult_census_income` | Download | Predict whether income exceeds $50K/yr based on census data |
| `amazon_employee_access_challenge` | Download | There is a considerable amount of data regarding an employee’s role within an organization and the r |
| `bnp_claims_management` | Download | The BNP Paribas Cardif Claims Management dataset. |
| `compas_recidivism` | HuggingFace | COMPAS recidivism risk prediction; criminal justice fairness benchmark |
| `credit_card_default` | HuggingFace | Credit card default prediction from payment history |
| `creditcard_fraud` | Kaggle | The Machine Learning Group ULB Dataset |
| `customer_churn_prediction` | Download | Dataset from a Kaggle competition that is about predicting whether a customer will change |
| `electricity_tabular` | HuggingFace | Electricity market price direction classification |
| `heart_failure` | HuggingFace | Heart failure clinical records; death event prediction |
| `higgs` | Download | The Higgs Boson dataset. |
| `ieee_fraud` | Download | The IEEE-CIS Fraud Detection Dataset |
| `imbalanced_insurance` | Kaggle | Health Insurance Cross Sell Prediction |
| `kdd_appetency` | Download | The KDD Cup 2009 Appetency dataset. |
| `kdd_churn` | Download | The KDD Cup 2009 Churn dataset. |
| `kdd_upselling` | Download | The KDD Cup 2009 Upselling dataset. |
| `noshow_appointments` | Kaggle | 110.527 medical appointments its 14 associated variables (characteristics). |
| `numerai28pt6` | Kaggle | Encrypted Stock Market Data from Numerai dataset from Kaggle. |
| `porto_seguro_safe_driver` | Download | Predict the probability that an auto insurance policy holder files a claim. |
| `santander_customer_satisfaction` | Download | Santander Customer Satisfaction Prediction. |
| `santander_customer_transaction` | Download | Santander Customer Transaction Prediction. |
| `synthetic_fraud` | Kaggle | The Synthetic Financial Datasets For Fraud Detection dataset. |
| `talkingdata_adtrack_fraud` | Download | TalkingData AdTracking Fraud Detection Challenge. |
| `telco_customer_churn` | Kaggle | The Telco customer churn data contains information about a fictional telco company |
| `titanic` | Download | The Titanic dataset: use machine learning to create a model |

### Tabular Regression (18 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `allstate_claims_severity` | Download | Allstate Claims Severity. |
| `ames_housing` | Download | The Ames Housing dataset. |
| `california_housing` | Download | California Housing dataset from the 1990 US Census. Predict median house value |
| `electricity` | Download | Electricity demand dataset. Half-hourly electricity demand in Victoria, Australia during 2014, along |
| `gaia_cepheids` | HuggingFace | Gaia DR3 Cepheids: predict stellar period from photometry. 15K examples. |
| `gaia_rrlyrae` | HuggingFace | Gaia DR3 RR Lyrae: predict fundamental period. 271K examples. |
| `gaia_spectroscopic_binaries` | HuggingFace | Gaia DR3 Spectroscopic Binaries: predict orbital period. 186K examples. |
| `gaia_young_stellar_objects` | HuggingFace | Gaia DR3 Young Stellar Objects: class prediction from photometry. 79K examples. |
| `mercedes_benz_greener` | Download | The Mercedes-Benz Greener Manufacturing dataset. |
| `naval` | Download | Condition Based Maintenance of Naval Propulsion Plants Data Set |
| `protein` | Download | Physicochemical Properties of Protein Tertiary Structure Data Set. |
| `repid` | HuggingFace | REPID: Retinal Perceptual Image Dataset — tabular perceptual quality metrics |
| `rossman_store_sales` | Download | The Rossmann Store Sales dataset. |
| `santander_value_prediction` | Download | The Santander Value Prediction Challenge dataset. |
| `sarcos` | Download | The data relates to an inverse dynamics problem for a seven |
| `stocks_daily_price` | HuggingFace | Daily stock OHLCV: predict close price from symbol and features. 25M examples. |
| `temperature` | Kaggle | Hourly temperature dataset from Kaggle |
| `yosemite` | Download | Yosemite temperatures dataset. |

### Multimodal (7 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `flickr8k` | Download | A new benchmark collection for sentence-based image description and search, |
| `goodbooks_books` | Download | goodbooks_books is a multimodal dataset of 10K books, taken from the goodreads dataset. |
| `insurance_lite` | Kaggle | The dataset consists of parameters such as the images of damaged cars, |
| `mobile_mold` | HuggingFace | MobileMold: smartphone food mold binary image classification. 3.5K examples. |
| `twitter_bots` | Kaggle | A dataset for Twitter Bot account detection. |
| `wmt15` | Kaggle | French/English parallel texts for training translation models. |
| `world_speech_asr` | HuggingFace | WorldSpeech: multilingual audio quality with CER measurement. 1.2M train examples. |

### Special / Utility (1 datasets)

| Dataset | Source | Description |
| ------- | ------ | ----------- |
| `hugging_face` | Download | Hugging Face Datasets |

## Adding datasets

To add a dataset to the Ludwig Dataset Zoo, see [Add a Dataset](../../developer_guide/add_a_dataset.md).