# NLP Models

## 1. Model Strategy

ResearchGPT deliberately demonstrates multiple generations of NLP techniques.

```text
Classical NLP
     ↓
TF-IDF
     ↓
Machine Learning
     ↓
Naive Bayes / Logistic Regression / SVM
     ↓
Deep Learning
     ↓
BiLSTM-CRF
     ↓
Transformers
     ↓
BERT / DistilBERT
     ↓
T5 / BART
     ↓
Sentence-BERT
     ↓
FAISS
     ↓
RAG
```

## 2. Classification

### TF-IDF + SVM

Primary classical baseline.

Pipeline:

```text
Text
 ↓
Preprocessing
 ↓
TF-IDF
 ↓
SVM
 ↓
Prediction
```

Additional baselines:

- Naive Bayes;
- Logistic Regression.

### Transformer Classification

Candidate family:

- BERT;
- DistilBERT.

The exact checkpoint must be recorded in experiment configuration.

## 3. NER

### BiLSTM-CRF

Pipeline:

```text
Tokens
 ↓
Embedding
 ↓
BiLSTM
 ↓
CRF
 ↓
Entity labels
```

### BERT Token Classification

Transformer alternative.

The tokenizer, checkpoint, label mapping, and maximum sequence length must be recorded.

## 4. Summarization

Candidate families:

- T5;
- BART.

The project should distinguish between extractive and abstractive behavior where relevant.

## 5. Semantic Embeddings

Sentence-BERT is the primary semantic embedding approach.

The initial project specification identifies `all-MiniLM-L6-v2` as the intended sentence-embedding model.

The exact model revision must be recorded when experiments are executed.

## 6. Question Answering

Candidate family:

- BERT;
- DistilBERT.

The initial QA system is intended to support extractive question answering over retrieved paper context.

## 7. Model Registry

When implemented, each model should have metadata:

```text
Model name:
Task:
Checkpoint:
Tokenizer:
Training dataset:
Dataset version:
Input format:
Output format:
Seed:
Hyperparameters:
Evaluation:
Artifact location:
Created at:
```

## 8. Model Selection Rule

Do not claim one model is superior without experimental evidence.

Comparisons must report the same evaluation protocol and clearly identify:

- dataset;
- split;
- metric;
- model;
- configuration.
