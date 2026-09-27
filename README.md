# ResearchGPT

> **AI Research Paper Assistant using NLP, Semantic Search, Question Answering and Retrieval-Augmented Generation**

ResearchGPT is an academic NLP system designed to help users upload research papers, understand their content, search them semantically, and ask questions grounded in the uploaded documents.

The project combines classical NLP, machine learning, deep learning, transformer models, sentence embeddings, vector search, and retrieval-augmented generation into one end-to-end research assistant.

## What ResearchGPT Does

```text
Research Paper PDF
       │
       ▼
 PDF Text Extraction
       │
       ▼
 Preprocessing
       │
       ├───────────────┬────────────────┬─────────────────┐
       ▼               ▼                ▼                 ▼
 Classification      NER          Summarization          Knowledge
       │               │                │
       └───────────────┴────────────────┘
                       │
                       ▼
                Paper Knowledge
                       │
                       ▼
                 Text Chunking
                       │
                       ▼
                 Sentence-BERT
                       │
                       ▼
                     FAISS
                       │
              ┌────────┴────────┐
              │                 │
        User Question           │
              ▼                 │
        Query Embedding         │
              ▼                 │
        Semantic Retrieval ─────┘
              │
              ▼
          QA / RAG
              │
              ▼
       Grounded Final Answer
```

## Core Capabilities

- Upload and process research-paper PDFs.
- Extract and clean paper text.
- Classify research papers into supported technical domains.
- Extract technical entities and keyphrases.
- Generate paper summaries.
- Create semantic embeddings for paper chunks.
- Search paper content using semantic similarity.
- Answer questions using retrieved paper context.
- Integrate retrieval with QA/RAG.
- Evaluate individual NLP components using task-appropriate metrics.

## NLP Components

| Component | Primary approach | Main evaluation |
|---|---|---|
| Research-paper classification | TF-IDF + SVM; transformer baseline | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Technical NER | BiLSTM-CRF; BERT token classification | Entity Precision, Recall, F1 |
| Summarization | T5/BART | ROUGE-1, ROUGE-2, ROUGE-L, optional BERTScore |
| Question Answering | BERT/DistilBERT | Exact Match, Token F1, Precision, Recall |
| Semantic retrieval | Sentence-BERT + FAISS | Retrieval/similarity evaluation |
| RAG | Retrieved context + QA/LLM layer | Groundedness and QA metrics |

## Experimental Datasets

The project uses benchmark datasets to train and evaluate individual NLP components. The final application is intended to process real research-paper PDFs.

- **IMDB** — binary sentiment classification
- **SMS Spam Collection** — binary spam classification
- **SST-2** — sentiment classification
- **CoNLL-2003** — named entity recognition
- **STS-B** — semantic textual similarity

See [`docs/Dataset.md`](docs/Dataset.md).

## Project Architecture

See [`docs/Architecture.md`](docs/Architecture.md).

## Development Workflow

ResearchGPT is developed one task at a time.

```text
Plan
  ↓
Implement
  ↓
Test
  ↓
Review
  ↓
Fix
  ↓
Retest
  ↓
Document
  ↓
Human approval
  ↓
Next task
```

No task is considered complete merely because the application runs.

See [`docs/Development.md`](docs/Development.md) and [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Repository Structure

```text
ResearchGPT/
├── .github/
│   └── workflows/
├── backend/
├── frontend/
├── models/
├── experiments/
├── data/
├── docs/
├── scripts/
├── tests/
├── .gitignore
├── LICENSE
├── PRD.md
├── PROJECT_STATUS.md
└── README.md
```

## Getting Started

The repository is being developed incrementally. The first implementation phase establishes the environment and project foundation before model training begins.

### Backend

```bash
cd backend
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies once `backend/requirements.txt` is populated by the approved foundation task:

```bash
pip install -r requirements.txt
```

### Frontend

The frontend will be initialized during the frontend implementation phase. Do not install a frontend stack independently of the documented architecture.

## Documentation

- [`PRD.md`](PRD.md) — product and academic requirements
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — current development state
- [`docs/Architecture.md`](docs/Architecture.md) — system architecture
- [`docs/Workflow.md`](docs/Workflow.md) — end-to-end application workflow
- [`docs/Dataset.md`](docs/Dataset.md) — datasets and reproducibility
- [`docs/NLP_Models.md`](docs/NLP_Models.md) — model strategy
- [`docs/Preprocessing.md`](docs/Preprocessing.md) — preprocessing pipeline
- [`docs/Semantic_Retrieval.md`](docs/Semantic_Retrieval.md) — embeddings and vector retrieval
- [`docs/RAG.md`](docs/RAG.md) — retrieval-augmented generation
- [`docs/Evaluation.md`](docs/Evaluation.md) — metrics and experiment reporting
- [`docs/Development.md`](docs/Development.md) — Antigravity development protocol

## Academic Positioning

The project demonstrates the progression:

```text
Classical NLP
    ↓
TF-IDF
    ↓
Machine Learning
    ↓
BiLSTM-CRF
    ↓
Transformers
    ↓
BERT / DistilBERT / T5 / BART
    ↓
Sentence Embeddings
    ↓
Semantic Search
    ↓
FAISS
    ↓
RAG
    ↓
Research Assistant
```

## Important Reproducibility Rule

No experiment result is added to the repository unless the dataset, split, model, tokenizer, preprocessing, seed, hyperparameters, evaluation metrics, environment, and execution status are recorded.

## Status

**Current state:** Repository initialization / planning.

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the authoritative status.
