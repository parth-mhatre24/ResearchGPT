# ResearchGPT Project Status

## Current State

- **Phase:** Phase 13 — Frontend User Interface
- **Current Task:** Task 19 (Frontend User Interface) — Complete & Verified
- **Status:** COMPLETED & VERIFIED (109 unit & integration tests passing with 0 errors, rich glassmorphic web UI verified across all 6 screens, awaiting human approval)
- **Completed Tasks:** 
  - Task 01 — Repository Audit (Approved)
  - Task 02 — Project Foundation (Approved)
  - Task 03 — PDF Upload (Approved)
  - Task 04 — PDF Extraction (Approved)
  - Task 05 — Preprocessing (Approved)
  - Task 06 — Dataset Pipeline & Download (Approved)
  - Task 07 — TF-IDF Classification (Approved)
  - Task 08 — Transformer Classification Service (Approved)
  - Task 09 — CRF NER Service (Approved)
  - Task 10 — BiLSTM-CRF NER Service & Benchmark (Approved)
  - Task 11 — Fine-Tuned BERT NER Service & Benchmark (Approved)
  - Task 12 — Abstractive Summarization Service & ROUGE Benchmark (Approved)
  - Task 13 — STS-B Semantic Similarity Service & Correlation Benchmark (Approved)
  - Task 14 — Document Chunking & Sentence-BERT Embedding Service (Approved)
  - Task 15 — Vector Indexing & Semantic Nearest-Neighbor Retrieval Service (Approved)
  - Task 16 — Extractive Question Answering Service & Benchmark (Approved)
  - Task 17 — Grounded RAG Pipeline Orchestrator with Source Citations (Approved)
  - Task 18 — Full System Integration & Production API Layer (Approved)
  - Task 19 — Frontend User Interface (Completed)
- **Approved Tasks:** Task 01, Task 02, Task 03, Task 04, Task 05, Task 06, Task 07, Task 08, Task 09, Task 10, Task 11, Task 12, Task 13, Task 14, Task 15, Task 16, Task 17, Task 18
- **Blocked Tasks:** None. Python 3.12 environment with PyTorch 2.5.1 +cu121 and Transformers 5.18.0 is fully operational. All 109 unit and integration tests pass cleanly.

## Progress Against PRD.md

- **PRD Functional Requirements:** **10 / 10 Completed & Approved (100%)**
  - [x] FR-01: PDF Upload (Completed & Approved)
  - [x] FR-02: PDF Text Extraction (Completed & Approved)
  - [x] FR-03: Preprocessing (Completed & Approved)
  - [x] FR-04: Classification (Completed & Approved)
  - [x] FR-05: Named Entity Recognition (Completed & Approved)
  - [x] FR-06: Summarization (Completed & Approved)
  - [x] FR-07: Semantic Similarity (Completed & Approved)
  - [x] FR-08: Semantic Retrieval (Completed & Approved)
  - [x] FR-09: Question Answering (Completed & Approved)
  - [x] FR-10: RAG Pipeline (Completed & Approved)
- **Task Sequence Progress:** **19 / 20 Implemented (95%)** (18 Approved, 1 Awaiting Approval)
- **Phases Progress:** **13 Completed / 14 Planned Phases (~93%)**

## Development Rules

1. **Single Active Task Rule**: Only one task may be active at a time.
2. **Standard Task Execution Cycle**:
   ```text
   Understand
   → Inspect
   → Implement
   → Test
   → Review
   → Fix
   → Retest
   → Document
   → Report
   → Human approval
   → Next task
   ```
3. **Mandatory Post-Completion Update Rule**:
   - `PROJECT_STATUS.md` **MUST be updated immediately after each completion of work** (upon finishing implementation, running all tests, and before proceeding to any subsequent task).
   - Every update must synchronize:
     - Current State (active phase, task, status);
     - Progress metrics against [PRD.md](PRD.md);
     - Planned Phases table and Task Sequence checklist;
     - Experiment results, architecture decisions, or known issues.
   - Do not mark a task as `Approved` until the project owner explicitly provides approval.
   - Never fabricate or assume metrics, benchmarks, or test results.

## Planned Phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Repository and project foundation | Completed & Approved |
| 2 | PDF upload and extraction | Completed & Approved |
| 3 | NLP preprocessing | Completed & Approved |
| 4 | Classification | Completed & Approved |
| 5 | NER | Completed & Approved |
| 6 | Summarization | Completed & Approved |
| 7 | Semantic similarity | Completed & Approved |
| 8 | Semantic embeddings | Completed & Approved |
| 9 | FAISS retrieval | Completed & Approved |
| 10 | Question answering | Completed & Approved |
| 11 | RAG | Completed & Approved |
| 12 | Full integration & Production API | Completed & Approved |
| 13 | Frontend User Interface | Completed — Rich Glassmorphism Web App & Static Integration |
| 14 | Final testing and hardening | Not started |

## Task Sequence

- [x] 1. Repository audit (Approved)
- [x] 2. Project foundation (Approved)
- [x] 3. PDF upload (Approved)
- [x] 4. PDF extraction (Approved)
- [x] 5. Preprocessing (Approved)
- [x] 6. Classification dataset pipeline & download (Approved)
- [x] 7. TF-IDF classification (Approved)
- [x] 8. Transformer classification (Approved)
- [x] 9. NER dataset pipeline + CRF NER baseline (Approved)
- [x] 10. BiLSTM-CRF NER (Approved)
- [x] 11. BERT NER (Approved)
- [x] 12. Summarization (Approved)
- [x] 13. STS-B semantic similarity (Approved)
- [x] 14. Paper chunking + Sentence-BERT (Approved)
- [x] 15. FAISS / Vector retrieval (Approved)
- [x] 16. QA (Approved)
- [x] 17. RAG (Approved)
- [x] 18. Full integration (Approved)
- [x] 19. Frontend (Implemented & Verified across all 6 screens)
- [ ] 20. Final testing

## Known Issues

- None. Python 3.12 environment with PyTorch 2.5.1 + Transformers 5.18.0 is fully operational. Test suite runs with 109 passing tests.

## Architecture Decisions

See [`docs/Architecture.md`](docs/Architecture.md).

## Dataset Acquisition Summary (Task 06)

All 5 benchmark datasets defined in `PRD.md` downloaded and staged in `data/raw/`:

| Dataset | Key | Task | Hub ID / Config | Splits & Row Counts | Staging Path |
|---|---|---|---|---|---|
| **IMDB Movie Reviews** | `imdb` | Binary sentiment | `stanfordnlp/imdb` | train (25,000), test (25,000) | `data/raw/imdb/` |
| **SMS Spam Collection** | `sms_spam` | Binary spam | `ucirvine/sms_spam` | train (4,459), val (557), test (558) | `data/raw/sms_spam/` |
| **SST-2** | `sst2` | Fine-grained/binary sentiment | `stanfordnlp/sst2` | train (67,349), val (872), test (1,821) | `data/raw/sst2/` |
| **CoNLL-2003** | `conll2003` | NER | `lhoestq/conll2003` | train (14,041), val (3,250), test (3,453) | `data/raw/conll2003/` |
| **STS-B** | `stsb` | Semantic textual similarity | `nyu-mll/glue` (`stsb`) | train (5,749), val (1,500), test (1,379) | `data/raw/stsb/` |

Central manifest generated at `data/raw/manifest.json`.
Programmatic loader implemented in `backend/app/services/dataset_service.py`.

---

## Experiment Results

### 1. Classical NLP Classification Baselines (Task 07)

Evaluated across all three benchmark datasets using TF-IDF feature representation (unigrams + bigrams, `max_features=20000`, `sublinear_tf=True`):

| Dataset | Classifier Model | Evaluation Split | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | F1 (Weighted) | Training Time |
|---|---|---|---|---|---|---|---|---|
| **SMS Spam** | Multinomial Naive Bayes | Test (558) | 96.24% | 97.93% | 85.71% | 90.80% | 95.97% | 0.11s |
| **SMS Spam** | Logistic Regression (L2) | Test (558) | 97.85% | 97.74% | 93.18% | 95.33% | 97.84% | 0.13s |
| **SMS Spam** | Support Vector Machine (LinearSVC) | Test (558) | **98.39%** | **97.87%** | **95.15%** | **96.43%** | **98.36%** | 0.12s |
| **IMDB** | Multinomial Naive Bayes | Test (25,000) | 85.62% | 85.63% | 85.62% | 85.62% | 85.62% | 10.23s |
| **IMDB** | Logistic Regression (L2) | Test (25,000) | **88.84%** | **88.84%** | **88.84%** | **88.84%** | **88.84%** | 9.27s |
| **IMDB** | Support Vector Machine (LinearSVC) | Test (25,000) | 87.48% | 87.48% | 87.48% | 87.48% | 87.48% | 9.40s |
| **SST-2** | Multinomial Naive Bayes | Val (872) | 78.56% | 78.68% | 78.43% | 78.37% | 78.41% | 0.70s |
| **SST-2** | Logistic Regression (L2) | Val (872) | **79.59%** | **79.57%** | **79.54%** | **79.54%** | **79.56%** | 0.80s |
| **SST-2** | Support Vector Machine (LinearSVC) | Val (872) | 77.87% | 77.87% | 77.80% | 77.79% | 77.81% | 1.09s |

*Trained models persisted to `models/saved/classical/` (`.joblib` + metadata JSON).*
*Detailed experiment logs with per-class metrics and confusion matrices recorded in `experiments/results/classification/`.*

---

### 2. Neural Sequence Tagging: BiLSTM-CRF NER (Task 10)

Evaluated on **CoNLL-2003** named entity recognition benchmark (14,041 train / 3,250 val / 3,453 test sentences) using word embeddings + BiLSTM + Linear-Chain CRF decoder:

| Metric | Validation Split (3,250 sentences) | Test Split (3,453 sentences) |
|---|:---:|:---:|
| **Micro F1** | **61.91%** | **56.90%** |
| **Micro Precision** | 79.85% | **76.12%** |
| **Micro Recall** | 50.56% | **45.43%** |
| **Macro F1** | 58.47% | **53.90%** |
| **Token Accuracy** | 91.47% | **90.29%** |

- **Per-Entity Test F1**: `LOC`: 68.13% | `PER`: 59.17% | `ORG`: 46.65% | `MISC`: 41.64%
- **Artifacts**: Saved to `models/saved/ner/bilstm_crf/` and `experiments/results/ner/bilstm_crf_results.json`.

---

### 3. Transformer Named Entity Recognition: BERT NER (Task 11)

Fine-tuned `dslim/bert-base-NER` on **CoNLL-2003** with subword token-to-word alignment:

| Metric | Validation Split (3,250 sentences) | Test Split (3,453 sentences) |
|---|:---:|:---:|
| **Micro F1** | **95.21%** | **91.48%** |
| **Micro Precision** | 95.17% | **91.11%** |
| **Micro Recall** | 95.24% | **91.86%** |
| **Macro F1** | 94.56% | **90.12%** |
| **Token Accuracy** | 99.14% | **98.23%** |

- **Per-Entity Test F1**: `PER`: **95.68%** | `LOC`: **93.38%** | `ORG`: **89.80%** | `MISC`: **81.60%**
- **Artifacts**: Saved to `models/saved/ner/bert/` and `experiments/results/ner/bert_results.json`.

---

### 4. Abstractive Document Summarization (Task 12)

Evaluated abstractive summarization using `t5-small` with beam search width = 4 on research document introductions against reference abstracts:

| Metric | Score | Target Standard |
|---|:---:|:---:|
| **ROUGE-1 F1** | **48.00%** | > 40.0% |
| **ROUGE-2 F1** | **21.53%** | > 18.0% |
| **ROUGE-L F1** | **25.65%** | > 22.0% |
| **Average Compression Ratio** | **43.6%** | 30% - 50% |

- **Artifacts**: Saved to `experiments/results/summarization/summarization_results.json`.

---

### 5. Semantic Textual Similarity: STS-B Benchmark (Task 13)

Evaluated on **STS-B** validation split (1,500 sentence pairs) comparing TF-IDF baseline vs. Sentence-BERT:

| Representation Method | Pearson Correlation ($r$) | Spearman Correlation ($\rho$) | Mean Squared Error (MSE) |
|---|:---:|:---:|:---:|
| **TF-IDF Cosine Baseline** | 0.6065 | 0.6350 | 2.8168 |
| **Sentence-BERT (`all-MiniLM-L6-v2`)** | **0.8709** | **0.8672** | **0.7875** |
| **Improvement ($\Delta$)** | **+0.2644 (+43.6%)** | **+0.2322 (+36.6%)** | **-2.0293 (-72.0%)** |

- **Artifacts**: Saved to `experiments/results/similarity/stsb_results.json`.

---

### 6. Vector Indexing & Semantic Retrieval (Tasks 14 & 15)

Evaluated dense chunking and vector nearest-neighbor retrieval on scientific research papers:

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim normalized vectors)
- **Index**: Exact Flat Inner Product (`IndexFlatIP_Cosine`)
- **Query Latency**: ~7.5 – 8.8 ms per research question
- **Artifacts**: Saved index matrix and metadata to `models/saved/retrieval_index/`.

---

### 7. Question Answering & Grounded RAG Pipeline (Tasks 16 & 17)

Evaluated extractive question answering and end-to-end grounded RAG:

- **QA Model**: `distilbert-base-cased-distilled-squad`
- **QA Metrics**: Exact Match: **100.00%** | Token F1: **100.00%** | Precision: **100.00%** | Recall: **100.00%**
- **RAG Latency**: ~120–130 ms per grounded response (Retrieval + QA span extraction)
- **Grounded Verification**: Answers questions from *Attention Is All You Need* and *BERT* with exact source citations and refuses out-of-domain questions with zero hallucination.
- **Artifacts**: Saved to `experiments/results/qa/qa_results.json`.
