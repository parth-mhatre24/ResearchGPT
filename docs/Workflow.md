# ResearchGPT Workflow

## 1. Application Workflow

```text
1. User uploads research paper
        ↓
2. Validate PDF
        ↓
3. Extract text and page metadata
        ↓
4. Preprocess document
        ↓
5. Run classification
        ↓
6. Run NER / technical entity extraction
        ↓
7. Generate summary
        ↓
8. Split paper into retrieval chunks
        ↓
9. Generate Sentence-BERT embeddings
        ↓
10. Store embeddings in FAISS
        ↓
11. User asks a question
        ↓
12. Embed question
        ↓
13. Search FAISS
        ↓
14. Retrieve relevant paper sections
        ↓
15. Run QA/RAG
        ↓
16. Return answer + source context
```

## 2. Development Workflow

Each implementation task follows:

```text
Read requirements
      ↓
Inspect repository
      ↓
Implement only current scope
      ↓
Add tests
      ↓
Run tests
      ↓
Review implementation
      ↓
Fix discovered issues
      ↓
Run tests again
      ↓
Update documentation/status
      ↓
Report completion
      ↓
Human approval
```

## 3. Development Phases

### Phase 1 — Foundation

- repository audit;
- Python environment;
- backend structure;
- testing;
- configuration.

### Phase 2 — Document Processing

- upload;
- validation;
- PDF extraction;
- page-aware document model.

### Phase 3 — Preprocessing

- classical NLP preprocessing;
- transformer-compatible preprocessing;
- shared interfaces.

### Phase 4 — Classification

- datasets;
- TF-IDF baselines;
- SVM;
- transformer classifier;
- evaluation.

### Phase 5 — NER

- CoNLL-2003 pipeline;
- BiLSTM-CRF;
- BERT token classification;
- entity evaluation.

### Phase 6 — Summarization

- dataset/model preparation;
- T5/BART inference;
- evaluation.

### Phase 7 — Semantic Similarity

- STS-B;
- baseline similarity;
- Sentence-BERT evaluation.

### Phase 8 — Retrieval

- chunking;
- embeddings;
- FAISS.

### Phase 9 — QA

- QA preprocessing;
- inference;
- evaluation.

### Phase 10 — RAG

- retrieval + QA integration;
- source metadata;
- grounded responses.

### Phase 11 — Full Integration

- model orchestration;
- API integration;
- document lifecycle.

### Phase 12 — Frontend

- upload UI;
- analysis;
- search;
- QA;
- source display.

### Phase 13 — Final Verification

- unit tests;
- integration tests;
- end-to-end tests;
- security checks;
- performance measurements;
- documentation audit.
