# ResearchGPT Project Status

## Current State

- **Phase:** Phase 4 — Classification
- **Current Task:** Task 07 — TF-IDF Classification
- **Status:** APPROVED
- **Completed Tasks:** Task 01 — Repository Audit, Task 02 — Project Foundation, Task 03 — PDF Upload, Task 04 — PDF Extraction, Task 05 — Preprocessing, Task 06 — Dataset Pipeline & Download, Task 07 — TF-IDF Classification
- **Approved Tasks:** Task 01 — Repository Audit, Task 02 — Project Foundation, Task 03 — PDF Upload, Task 04 — PDF Extraction, Task 05 — Preprocessing, Task 06 — Dataset Pipeline & Download, Task 07 — TF-IDF Classification
- **Blocked Tasks:** None

## Progress Against PRD.md

- **PRD Functional Requirements:** **3 / 10 Completed (30%)**
  - [x] FR-01: PDF Upload (Completed & Approved)
  - [x] FR-02: PDF Text Extraction (Completed & Approved)
  - [x] FR-03: Preprocessing (Completed & Approved)
  - [ ] FR-04: Classification (Classical ML & Transformer) — Classical baselines completed; Transformer classification pending (Task 08)
  - [ ] FR-05: Named Entity Recognition / Keyphrase Extraction (BiLSTM-CRF & BERT)
  - [ ] FR-06: Summarization (T5 / BART)
  - [ ] FR-07: Semantic Similarity (STS-B)
  - [ ] FR-08: Semantic Retrieval (Sentence-BERT + FAISS)
  - [ ] FR-09: Question Answering (BERT / DistilBERT QA)
  - [ ] FR-10: RAG Pipeline (Retrieved Paper Grounding)
- **Task Sequence Progress:** **7 / 20 Completed (35%)** (6 Approved, 1 Awaiting Approval)
- **Phases Progress:** **3 Approved + Phase 4 In Progress / 14 Planned Phases (~25%)**

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
| 4 | Classification | In Progress |
| 5 | NER | Not started |
| 6 | Summarization | Not started |
| 7 | Semantic similarity | Not started |
| 8 | Semantic embeddings | Not started |
| 9 | FAISS retrieval | Not started |
| 10 | Question answering | Not started |
| 11 | RAG | Not started |
| 12 | Full integration | Not started |
| 13 | Frontend | Not started |
| 14 | Final testing and hardening | Not started |

## Task Sequence

- [x] 1. Repository audit (Approved)
- [x] 2. Project foundation (Approved)
- [x] 3. PDF upload (Approved)
- [x] 4. PDF extraction (Approved)
- [x] 5. Preprocessing (Approved)
- [x] 6. Classification dataset pipeline & download (Approved)
- [x] 7. TF-IDF classification (Completed, Awaiting Approval)
- [ ] 8. Transformer classification
- [ ] 9. NER dataset pipeline
- [ ] 10. BiLSTM-CRF NER
- [ ] 11. BERT NER
- [ ] 12. Summarization
- [ ] 13. STS-B semantic similarity
- [ ] 14. Paper chunking + Sentence-BERT
- [ ] 15. FAISS
- [ ] 16. QA
- [ ] 17. RAG
- [ ] 18. Full integration
- [ ] 19. Frontend
- [ ] 20. Final testing

## Known Issues

None recorded yet.

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

## Experiment Results

### Classical NLP Classification Baselines (Task 07)

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
