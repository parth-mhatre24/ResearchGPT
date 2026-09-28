# ResearchGPT — Product Requirements Document

## 1. Project Overview

ResearchGPT is an AI-powered research-paper assistant that accepts research papers in PDF format, extracts their content, applies multiple independently trained NLP components, creates a searchable semantic representation, and answers user questions using retrieved paper context.

The project is both:

1. an academic NLP/ML experimentation project, and
2. an integrated research-paper assistant application.

## 2. Problem Statement

Research papers are information-dense and often require users to manually locate algorithms, datasets, technologies, metrics, findings, and relevant passages.

ResearchGPT aims to reduce this manual effort by combining:

- document processing,
- research-domain classification,
- technical entity extraction,
- summarization,
- semantic retrieval,
- question answering,
- retrieval-augmented generation.

## 3. Goals

### Primary goals

- Accept research-paper PDFs.
- Extract usable text while preserving source/page metadata where available.
- Classify papers into supported research domains.
- Extract technical entities and keyphrases.
- Generate summaries.
- Create semantic embeddings of document chunks.
- Retrieve relevant paper sections for a question.
- Answer questions using retrieved evidence.
- Evaluate each NLP component independently.
- Integrate the components into one usable application.

### Academic goals

Compare classical NLP/ML approaches with deep-learning and transformer approaches.

The project should demonstrate a progression from:

```text
TF-IDF / classical ML
→ BiLSTM-CRF
→ Transformers
→ Sentence-BERT
→ FAISS
→ RAG
```

## 4. Supported Research Domains

The planned research-paper classification categories are:

- Cybersecurity
- Artificial Intelligence
- Natural Language Processing
- Cloud Computing
- Internet of Things
- Blockchain
- Computer Vision
- Robotics

The training benchmark datasets listed below are not themselves equivalent to these eight research-paper categories. They are experimental datasets for evaluating classification techniques.

## 5. Functional Requirements

### FR-01 — PDF Upload

The application shall allow a user to upload a research-paper PDF.

Requirements:

- accept PDF files;
- validate file type;
- apply reasonable file-size limits;
- avoid unsafe file paths;
- store uploaded files safely;
- expose clear errors.

### FR-02 — PDF Text Extraction

The system shall extract text from uploaded PDFs.

The extraction layer should preserve, where possible:

- page number;
- document identifier;
- section/chunk boundaries;
- source text.

### FR-03 — Preprocessing

The classical NLP pipeline may include:

- sentence segmentation;
- tokenization;
- lowercasing;
- stopword removal;
- lemmatization.

Transformer pipelines should use lighter preprocessing and retain the text needed by the tokenizer.

### FR-04 — Classification

The project shall experiment with:

- TF-IDF + Naive Bayes;
- TF-IDF + Logistic Regression;
- TF-IDF + SVM;
- BERT/DistilBERT-style transformer classification.

Evaluation:

- Accuracy;
- Precision;
- Recall;
- F1;
- Confusion Matrix.

### FR-05 — Named Entity Recognition / Keyphrase Extraction

The NER component shall target technical entities such as:

- algorithms;
- datasets;
- technologies;
- metrics;
- organizations;
- tools.

The primary neural approach is BiLSTM-CRF, with BERT token classification as a transformer alternative.

Evaluation:

- entity-level Precision;
- entity-level Recall;
- entity-level F1.

### FR-06 — Summarization

The system shall provide research-paper summarization using a T5/BART-family approach.

Evaluation:

- ROUGE-1;
- ROUGE-2;
- ROUGE-L;
- optional BERTScore.

### FR-07 — Semantic Similarity

The project shall evaluate semantic textual similarity using STS-B and compare classical similarity representations with Sentence-BERT embeddings where appropriate.

Possible metrics:

- Pearson correlation;
- Spearman correlation;
- MSE where appropriate.

### FR-08 — Semantic Retrieval

The system shall:

1. split papers into meaningful chunks;
2. encode chunks with Sentence-BERT;
3. store embeddings in FAISS;
4. encode user queries;
5. retrieve relevant chunks using vector similarity.

### FR-09 — Question Answering

The system shall provide extractive/question-answering capability using a BERT/DistilBERT-family QA model.

Evaluation:

- Exact Match;
- Token F1;
- Precision;
- Recall.

### FR-10 — RAG

The final answer pipeline shall use retrieved paper content as context.

```text
Question
→ Query embedding
→ FAISS retrieval
→ Relevant paper sections
→ QA / RAG generation
→ Grounded answer
```

The final system should preserve enough source metadata to identify where an answer came from.

## 6. Non-Functional Requirements

### Reproducibility

Experiments must record:

- dataset;
- dataset version/source;
- split;
- random seed;
- preprocessing;
- tokenizer;
- model;
- hyperparameters;
- training configuration;
- evaluation metrics;
- environment.

### Maintainability

The backend, frontend, model implementations, experiments, and tests should remain separated.

### Security

The application should:

- validate uploads;
- avoid arbitrary file paths;
- avoid committing secrets;
- avoid executing uploaded content;
- keep temporary files controlled;
- avoid logging sensitive document contents unnecessarily.

### Performance

Performance optimization should not be introduced prematurely. First establish correct behavior and measurable baselines.

## 7. Dataset Requirements

The project uses the following benchmark datasets to train, benchmark, and evaluate individual NLP components before integrating them into the end-to-end research assistant:

| # | Dataset | Task | Link | Why it fits |
|---|---|---|---|---|
| 1 | **IMDB Movie Reviews** | Binary sentiment classification | [Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb) | Directly replicates Paper 1's setup; large, clean, well-benchmarked. |
| 2 | **SMS Spam Collection** | Binary spam classification | [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | Small, fast to train on; good for testing TF-IDF vs BERT on short text. |
| 3 | **SST-2 (Stanford Sentiment Treebank)** | Fine-grained/binary sentiment | [Hugging Face](https://huggingface.co/datasets/stanfordnlp/sst2) | Part of GLUE; lets you benchmark against BERT/DistilBERT paper results directly. |
| 4 | **CoNLL-2003** | Named Entity Recognition | [Hugging Face](https://huggingface.co/datasets/eriktks/conll2003) | Covers the NER angle (Paper 2 - Lample et al.); adds task diversity beyond classification. |
| 5 | **STS Benchmark (STS-B)** | Semantic textual similarity | [STS Wiki](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) | Lets you test Sentence-BERT style embeddings; nicely closes the loop with Paper 6. |

### Dataset Utilization Notes

- **IMDB Movie Reviews & SST-2**: Establish baselines and comparative benchmarks between classical ML models (TF-IDF + Naive Bayes/SVM/Logistic Regression) and Transformer-based models (BERT / DistilBERT).
- **SMS Spam Collection**: Provides a short-text classification benchmark to observe feature sparsity challenges with TF-IDF versus contextual token embeddings in BERT.
- **CoNLL-2003**: Serves as the primary sequence labeling benchmark to evaluate BiLSTM-CRF against transformer-based token classification.
- **STS Benchmark (STS-B)**: Evaluates semantic similarity representations (cosine similarity with TF-IDF / GloVe vs. bi-encoder Sentence-BERT embeddings) before retrieval indexing with FAISS.

*Note: The final application is intended to accept research papers independently of these benchmark datasets.*

## 8. Model Requirements

| Component | Primary model | Alternative / baseline |
|---|---|---|
| Classification | SVM / DistilBERT | NB, Logistic Regression |
| NER | BiLSTM-CRF | BERT token classification |
| Summarization | T5/BART | baseline only if needed |
| Semantic retrieval | Sentence-BERT | TF-IDF similarity baseline |
| QA | BERT/DistilBERT | alternative transformer checkpoint |
| Vector search | FAISS | exact similarity baseline for testing |

## 9. Success Criteria

The project is considered complete when:

- the four major NLP components are implemented;
- each component has tests;
- experiments have reproducible configurations;
- evaluation results are recorded without fabricated values;
- PDF processing works;
- semantic retrieval works;
- QA/RAG is grounded in retrieved paper content;
- frontend and backend communicate correctly;
- integration tests pass;
- final documentation matches the implemented system.

## 10. Out of Scope Until Explicitly Approved

- production-scale distributed training;
- autonomous web research;
- unrestricted external knowledge generation;
- committing large datasets to Git;
- committing large model binaries without an explicit artifact strategy;
- unrelated product features.

## 11. Source of Requirements

The project specification supplied for ResearchGPT is the authoritative academic basis for this PRD. It describes four independently trained NLP components, Sentence-BERT, FAISS, RAG, preprocessing, evaluation, and the end-to-end architecture.
