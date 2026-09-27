# Semantic Retrieval

## 1. Purpose

Semantic retrieval allows ResearchGPT to find paper sections that are conceptually relevant to a user's question rather than relying only on exact keyword matches.

## 2. Retrieval Pipeline

```text
Research Paper
      ↓
Text Chunks
      ↓
Sentence-BERT
      ↓
Dense Embeddings
      ↓
FAISS Index
```

At query time:

```text
User Question
      ↓
Sentence-BERT
      ↓
Query Embedding
      ↓
FAISS Similarity Search
      ↓
Top-k Chunks
      ↓
QA / RAG
```

## 3. Embedding Model

The initial project specification identifies:

```text
all-MiniLM-L6-v2
```

as the intended Sentence-BERT embedding model.

The exact checkpoint/version must be recorded in experiment metadata.

## 4. Chunk Metadata

Every indexed chunk should retain:

```text
chunk_id
document_id
page_start
page_end
text
embedding/index identifier
```

Additional section metadata may be retained when extraction makes it reliable.

## 5. FAISS

FAISS is the initial vector-search engine.

The first implementation should prioritize correctness and reproducibility before introducing approximate indexes or performance optimizations.

## 6. Retrieval Parameters

At minimum, experiments should record:

- embedding model;
- index type;
- similarity metric;
- top-k;
- chunk size;
- chunk overlap;
- preprocessing;
- dataset/document set.

## 7. Retrieval Quality

Retrieval should eventually be tested using labeled or manually validated query/document pairs where available.

Do not claim retrieval quality without a defined evaluation protocol.

## 8. Source Grounding

Retrieved chunks must remain traceable to their original document and page.

The retrieval layer should return both:

1. the text used as context;
2. metadata explaining where it came from.
