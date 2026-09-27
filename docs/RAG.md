# Retrieval-Augmented Generation

## 1. Purpose

ResearchGPT uses retrieval to provide relevant paper context before answering a user's question.

The goal is to reduce unsupported answers by grounding the answer pipeline in retrieved document content.

## 2. RAG Pipeline

```text
User Question
      ↓
Query Embedding
      ↓
FAISS Search
      ↓
Relevant Paper Chunks
      ↓
Context Assembly
      ↓
QA / RAG Model
      ↓
Answer
      +
Source Metadata
```

## 3. Context Construction

The RAG layer should:

- retrieve a controlled number of chunks;
- preserve document and page metadata;
- avoid exceeding the downstream model's context limit;
- avoid mixing unrelated documents without an explicit multi-document mode.

## 4. Grounding Rule

The final answer should be based on retrieved paper context.

If the required information is not found, the system should communicate that the paper context does not contain enough evidence rather than silently inventing a paper-specific answer.

## 5. Source Display

Where practical, the UI should display:

- document name;
- page number;
- relevant excerpt;
- retrieval score or rank when useful.

## 6. Multi-Paper Retrieval

Multi-paper retrieval may be supported after the single-paper pipeline is stable.

When multiple documents are indexed, each retrieved chunk must retain its document identity.

## 7. Hallucination Testing

RAG testing should include questions where:

- the answer is clearly present;
- the answer requires combining two retrieved sections;
- the answer is absent;
- similar but incorrect terminology appears;
- the paper contains conflicting statements.

## 8. RAG Evaluation

The project should report objective QA metrics where a labeled dataset is available and should separately report retrieval/context quality.

Do not use a subjective claim such as "hallucination-free."
