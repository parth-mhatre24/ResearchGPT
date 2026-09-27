# ResearchGPT Dataset Strategy

## 1. Purpose

The benchmark datasets are used to train and evaluate individual NLP capabilities.

They are not intended to replace the final application's real-world input: research papers in PDF format.

## 2. Dataset Table

| Dataset | Task | Intended component |
|---|---|---|
| IMDB | Binary sentiment classification | Classification |
| SMS Spam Collection | Binary spam classification | Classification |
| SST-2 | Sentiment classification | Classification |
| CoNLL-2003 | Named Entity Recognition | NER |
| STS-B | Semantic Textual Similarity | Semantic similarity |

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
