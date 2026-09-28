# ResearchGPT Dataset Strategy

## 1. Purpose

The benchmark datasets are used to train and evaluate individual NLP capabilities.

They are not intended to replace the final application's real-world input: research papers in PDF format.

## 2. Dataset Table

| # | Dataset | Task | Link | Why it fits | Intended component |
|---|---|---|---|---|---|
| 1 | **IMDB Movie Reviews** | Binary sentiment classification | [Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb) | Directly replicates Paper 1's setup; large, clean, well-benchmarked | Classification |
| 2 | **SMS Spam Collection** | Binary spam classification | [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | Small, fast to train on; good for testing TF-IDF vs BERT on short text | Classification |
| 3 | **SST-2 (Stanford Sentiment Treebank)** | Fine-grained/binary sentiment | [Hugging Face](https://huggingface.co/datasets/stanfordnlp/sst2) | Part of GLUE; lets you benchmark against BERT/DistilBERT paper results directly | Classification |
| 4 | **CoNLL-2003** | Named Entity Recognition | [Hugging Face](https://huggingface.co/datasets/eriktks/conll2003) | Covers the NER angle (Paper 2 - Lample et al.); adds task diversity beyond classification | NER |
| 5 | **STS Benchmark (STS-B)** | Semantic textual similarity | [STS Wiki](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) | Lets you test Sentence-BERT style embeddings; nicely closes the loop with Paper 6 | Semantic similarity |

## 3. Classification Datasets

### IMDB

Use for binary sentiment classification experiments.

Record:

- source identifier;
- dataset version;
- train/test split;
- preprocessing;
- model;
- seed;
- metrics.

### SMS Spam Collection

Use for binary spam classification experiments.

The project should treat it as a separate text-classification benchmark rather than as a research-domain classifier.

### SST-2

Use for sentence-level sentiment classification experiments.

## 4. NER Dataset

### CoNLL-2003

Use for sequence labeling and NER experiments.

The project should report entity-level evaluation rather than only token accuracy.

## 5. Semantic Similarity Dataset

### STS-B

Use for semantic textual similarity.

The experiment should compare an appropriate classical representation/baseline with Sentence-BERT embeddings.

## 6. Dataset Reproducibility Record

Every experiment should record:

```text
Dataset name:
Source:
Version:
Download date:
License:
Split:
Preprocessing:
Random seed:
Number of samples:
Model:
Tokenizer:
Training configuration:
Evaluation metrics:
Environment:
```

## 7. Data Leakage

Do not:

- fit a vocabulary on test data;
- tune hyperparameters using test labels;
- mix train and test records;
- generate retrieval indexes from evaluation data when the experiment is intended to be held out.

## 8. Git Policy

Large datasets should not be committed directly to the repository.

`data/README.md` should document where data comes from and how to obtain it.

## 9. Research-Paper Data

Uploaded research papers belong to the application's document-processing workflow and should not automatically become training data.

Do not add user-uploaded documents to model-training datasets without explicit authorization and a documented data-governance decision.
