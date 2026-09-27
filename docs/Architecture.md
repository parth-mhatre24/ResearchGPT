# ResearchGPT Architecture

## 1. Architectural Goal

ResearchGPT separates document ingestion, NLP analysis, semantic indexing, retrieval, question answering, and presentation.

The architecture is intentionally modular so each NLP component can be trained, evaluated, replaced, and tested independently.

## 2. High-Level Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                     ResearchGPT UI                       │
│       Upload PDF · Analyze · Search · Ask Question       │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                     Backend API                          │
│ Upload · Documents · Analysis · Search · QA              │
└────────────────────────────┬─────────────────────────────┘
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
┌──────────────────┐ ┌────────────────┐ ┌─────────────────┐
│ Document Service │ │ NLP Services   │ │ Retrieval       │
│ PDF extraction   │ │ Classification │ │ Chunking        │
│ metadata         │ │ NER            │ │ SBERT           │
│ pages            │ │ Summarization  │ │ FAISS           │
└────────┬─────────┘ │ QA             │ └────────┬────────┘
         │           └───────┬────────┘          │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ RAG / Answering │
                    │ Retrieved       │
                    │ context + QA    │
                    └────────┬────────┘
                             ▼
                    Grounded response
```

## 3. Document Flow

```text
PDF
 ↓
Validation
 ↓
Text Extraction
 ↓
Page-aware document representation
 ↓
Preprocessing
 ↓
Analysis
 ↓
Chunking
 ↓
Embedding
 ↓
Vector index
 ↓
Question retrieval
 ↓
QA/RAG
```

## 4. Component Responsibilities

### Frontend

Responsible for:

- PDF upload;
- processing status;
- classification display;
- extracted entities;
- summary display;
- semantic search;
- question answering;
- source/context display.

### Backend API

Responsible for:

- request validation;
- file handling;
- orchestration;
- model-service calls;
- retrieval;
- response formatting.

### Document Service

Responsible for:

- PDF validation;
- extraction;
- page metadata;
- normalized document representation.

### Classification Service

Responsible for loading the approved classifier and returning:

- predicted category;
- confidence/probability when supported;
- model metadata.

### NER Service

Responsible for:

- technical entity extraction;
- entity labels;
- character/token offsets where practical.

### Summarization Service

Responsible for:

- summary generation;
- length/configuration handling;
- source association.

### Embedding Service

Responsible for:

- chunk embedding;
- query embedding;
- model version tracking.

### Retrieval Service

Responsible for:

- FAISS index management;
- top-k retrieval;
- similarity scores;
- source metadata.

### QA/RAG Service

Responsible for:

- receiving user question;
- retrieving relevant context;
- running QA/generation;
- returning source-aware answers.

## 5. Data Contracts

A document should conceptually contain:

```text
Document
├── document_id
├── filename
├── metadata
└── pages[]
    ├── page_number
    └── text
```

A chunk should contain:

```text
Chunk
├── chunk_id
├── document_id
├── page_start
├── page_end
├── text
└── embedding_id
```

A retrieval result should contain:

```text
RetrievedChunk
├── chunk_id
├── text
├── similarity_score
├── document_id
└── page metadata
```

## 6. Model Independence

Each major NLP model must be independently testable.

The application should not make the frontend responsible for model internals.

## 7. Source Grounding

Final answers should retain a connection to retrieved document chunks.

A response should not silently claim that information came from the paper if the information was not present in the retrieved context.

## 8. Architecture Change Policy

A change that affects multiple components requires documentation in the relevant architecture document before implementation is considered complete.

Do not silently replace a primary algorithm with a different architecture.

## 9. Initial Architecture Decisions

- Sentence-BERT is the semantic embedding layer.
- FAISS is the initial vector retrieval engine.
- TF-IDF is the classical text representation baseline.
- SVM is the primary classical classification model.
- BiLSTM-CRF is the primary classical/deep-learning NER approach.
- BERT-family token classification is the transformer NER alternative.
- T5/BART is the summarization family.
- BERT/DistilBERT is the QA family.
