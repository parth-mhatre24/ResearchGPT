# ResearchGPT — Empirical Benchmark Results

## Overview & Repository Status

All core natural language processing engines, neural sequence taggers, transformer fine-tuning pipelines, abstractive summarizers, sentence embedding encoders, vector retrieval search engines, extractive question answering models, and grounded RAG pipelines have been built, verified, and benchmarked on real-world reference datasets.

- **Total Unit & Integration Test Suite**: **109 tests passed, 0 errors, 0 skipped** (`pytest tests/ -v`)
- **Execution Environment**: Python 3.12.10 | PyTorch 2.5.1+cu121 | Hugging Face Transformers 5.18.0

---

## 1. Classical Text Classification (Task 07)

Evaluated across all three benchmark datasets using unigrams + bigrams (`max_features=20000`, `sublinear_tf=True`):

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

*Key Findings*:
- On short text (`sms_spam`), **LinearSVC** achieved top performance (**98.39% accuracy / 96.43% macro F1**).
- On long document text (`imdb`), **Logistic Regression** led with **88.84% accuracy / F1**, directly replicating classical baseline targets (~88%).
- On sentence-level sentiment (`sst2`), **Logistic Regression** achieved **79.59% accuracy**.

---

## 2. Neural Sequence Tagging: BiLSTM-CRF NER (Task 10)

Evaluated on the **CoNLL-2003** named entity recognition benchmark (14,041 train / 3,250 val / 3,453 test sentences) using word embeddings + Bidirectional LSTM + Linear-Chain CRF decoder:

- **Epochs**: 3 | **Vocabulary Size**: 21,011 tokens | **Embedding Dim**: 100 | **Hidden Dim**: 256

| Metric | Validation Split (3,250 sentences) | Test Split (3,453 sentences) |
|---|:---:|:---:|
| **Micro F1** | **61.91%** | **56.90%** |
| **Micro Precision** | 79.85% | **76.12%** |
| **Micro Recall** | 50.56% | **45.43%** |
| **Macro F1** | 58.47% | **53.90%** |
| **Token Accuracy** | 91.47% | **90.29%** |

### Per-Entity Test Breakdown (CoNLL-2003 Test):
- **LOC (Locations)**: Precision: 77.77% | Recall: 60.61% | **F1: 68.13%** (Support: 1,668)
- **PER (Persons)**: Precision: 74.00% | Recall: 49.29% | **F1: 59.17%** (Support: 1,617)
- **ORG (Organizations)**: Precision: 78.91% | Recall: 33.11% | **F1: 46.65%** (Support: 1,661)
- **MISC (Miscellaneous)**: Precision: 70.03% | Recall: 29.63% | **F1: 41.64%** (Support: 702)

*Artifacts*:
- Model Weights: [`models/saved/ner/bilstm_crf/model.pt`](file:///c:/Pinak/Project/NLP/ResearchGPT/models/saved/ner/bilstm_crf/model.pt)
- Metadata & Vocab: [`models/saved/ner/bilstm_crf/model_meta.json`](file:///c:/Pinak/Project/NLP/ResearchGPT/models/saved/ner/bilstm_crf/model_meta.json)
- Result Metrics: [`experiments/results/ner/bilstm_crf_results.json`](file:///c:/Pinak/Project/NLP/ResearchGPT/experiments/results/ner/bilstm_crf_results.json)

---

## 3. Transformer Named Entity Recognition: BERT NER (Task 11)

Fine-tuned `dslim/bert-base-NER` on **CoNLL-2003** with subword token-to-word alignment and masked label sequence (`-100`):

- **Epochs**: 3 | **Learning Rate**: 2e-5 | **Batch Size**: 16 | **Optimizer**: AdamW

| Metric | Validation Split (3,250 sentences) | Test Split (3,453 sentences) |
|---|:---:|:---:|
| **Micro F1** | **95.21%** | **91.48%** |
| **Micro Precision** | 95.17% | **91.11%** |
| **Micro Recall** | 95.24% | **91.86%** |
| **Macro F1** | 94.56% | **90.12%** |
| **Token Accuracy** | 99.14% | **98.23%** |

### Per-Entity Test Breakdown (CoNLL-2003 Test):
- **PER (Persons)**: Precision: 96.25% | Recall: 95.11% | **F1: 95.68%** (Support: 1,617)
- **LOC (Locations)**: Precision: 93.72% | Recall: 93.05% | **F1: 93.38%** (Support: 1,668)
- **ORG (Organizations)**: Precision: 88.44% | Recall: 91.21% | **F1: 89.80%** (Support: 1,661)
- **MISC (Miscellaneous)**: Precision: 80.19% | Recall: 83.05% | **F1: 81.60%** (Support: 702)

*Artifacts*:
- Saved Model & Tokenizer: [`models/saved/ner/bert/`](file:///c:/Pinak/Project/NLP/ResearchGPT/models/saved/ner/bert)
- Result Metrics: [`experiments/results/ner/bert_results.json`](file:///c:/Pinak/Project/NLP/ResearchGPT/experiments/results/ner/bert_results.json)

---

## 4. NER Comparative Analysis (CoNLL-2003 Test Split)

| Architecture | Micro F1 | Precision | Recall | Token Accuracy | LOC F1 | PER F1 | ORG F1 | MISC F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BiLSTM-CRF** (Task 10) | 56.90% | 76.12% | 45.43% | 90.29% | 68.13% | 59.17% | 46.65% | 41.64% |
| **BERT NER** (Task 11) | **91.48%** | **91.11%** | **91.86%** | **98.23%** | **93.38%** | **95.68%** | **89.80%** | **81.60%** |
| **Improvement (BERT vs BiLSTM-CRF)** | **+34.58%** | **+14.99%** | **+46.43%** | **+7.94%** | **+25.25%** | **+36.51%** | **+43.15%** | **+39.96%** |

*Insight*: Pre-trained contextual bidirectional representations (BERT) dramatically resolve long-tail entity span boundaries and ambiguous named entities (especially ORG and MISC) compared to un-pretrained word embedding BiLSTM-CRF models.

---

## 5. Abstractive Document Summarization (Task 12)

Evaluated abstractive summarization using `t5-small` with beam search width = 4 on landmark scientific paper introductions and abstracts:

- **Model**: `t5-small`
- **Beam Search Width**: 4
- **Length Constraint**: Min 20 tokens, Max 150 tokens

| Metric | Score | Target Standard |
|---|:---:|:---:|
| **ROUGE-1 F1** | **48.00%** | > 40.0% |
| **ROUGE-2 F1** | **21.53%** | > 18.0% |
| **ROUGE-L F1** | **25.65%** | > 22.0% |
| **Average Compression Ratio** | **43.6%** | 30% - 50% |

### Qualitative Summarization Samples:
1. **Attention Is All You Need**:
   - *Input Length*: 80 words
   - *Generated Summary*: `"the dominant sequence transduction models are based on complex recurrent or convolutional neural networks. the best performing models connect the encoder and decoder through an attention mechanism. we propose a new simple network architecture, the Transformer."` (36 words, 45.0% compression)
2. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**:
   - *Input Length*: 78 words
   - *Generated Summary*: `"a new language representation model called BERT is designed to pre-train deep bidirectional representations from unlabeled text. the pre-trained model can be fine-tuned with just one additional output layer to create state-of-the-art models."` (33 words, 42.3% compression)

*Artifacts*:
- Result Metrics: [`experiments/results/summarization/summarization_results.json`](file:///c:/Pinak/Project/NLP/ResearchGPT/experiments/results/summarization/summarization_results.json)

---

## 6. Semantic Textual Similarity: STS-B Benchmark (Task 13)

Evaluated on the **STS Benchmark (STS-B)** validation split (1,500 sentence pairs) comparing lexical overlap (TF-IDF Cosine) with dense semantic representation (Sentence-BERT):

- **Sentence-BERT Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Evaluation Split**: Validation (1,500 pairs)

| Representation Method | Pearson Correlation ($r$) | Spearman Correlation ($\rho$) | Mean Squared Error (MSE) |
|---|:---:|:---:|:---:|
| **TF-IDF Cosine Baseline** | 0.6065 | 0.6350 | 2.8168 |
| **Sentence-BERT (`all-MiniLM-L6-v2`)** | **0.8709** | **0.8672** | **0.7875** |
| **Improvement ($\Delta$)** | **+0.2644 (+43.6%)** | **+0.2322 (+36.6%)** | **-2.0293 (-72.0%)** |

*Insight*: Sentence-BERT captures nuanced synonymy and compositional semantics that bag-of-words / TF-IDF completely misses, achieving an outstanding **0.8709 Pearson correlation** on STS-B.

*Artifacts*:
- Result Metrics: [`experiments/results/similarity/stsb_results.json`](file:///c:/Pinak/Project/NLP/ResearchGPT/experiments/results/similarity/stsb_results.json)

---

## 7. Vector Indexing & Semantic Retrieval (Tasks 14 & 15)

Evaluated dense document chunking and vector nearest-neighbor retrieval on landmark scientific papers (*Attention Is All You Need* and *BERT*):

- **Chunking Strategy**: Sliding window (50 tokens, 10 overlap) with sentence-boundary preservation
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim normalized)
- **Index Type**: Exact Flat Inner Product (`IndexFlatIP_Cosine`)
- **Average Query Latency**: **~7.5 – 8.8 milliseconds**

### Semantic Query Verification:
1. **Query**: *"How does Scaled Dot-Product Attention compute weights between query and key vectors?"*
   - *Top-1 Retrieved*: `paper_attention_2017` (Score: **0.6550**)
   - *Latency*: 8.85 ms
2. **Query**: *"What pre-training technique does BERT use to learn bidirectional representations?"*
   - *Top-1 Retrieved*: `paper_bert_2018` (Score: **0.7518**)
   - *Latency*: 8.63 ms
3. **Query**: *"Why is the Transformer more parallelizable than recurrent neural networks?"*
   - *Top-1 Retrieved*: `paper_attention_2017` (Score: **0.6875**)
   - *Latency*: 6.86 ms

*Artifacts*:
- Vector Matrix: [`models/saved/retrieval_index/vectors.npy`](file:///c:/Pinak/Project/NLP/ResearchGPT/models/saved/retrieval_index/vectors.npy)
- Index Metadata: [`models/saved/retrieval_index/metadata.json`](file:///c:/Pinak/Project/NLP/ResearchGPT/models/saved/retrieval_index/metadata.json)

---

## 8. Extractive Question Answering Engine (Task 16)

Evaluated Transformer Question Answering model (`distilbert-base-cased-distilled-squad`) on scientific research question answering pairs:

- **Model**: `distilbert-base-cased-distilled-squad`
- **Inference Latency**: ~30–45 ms per question

| Metric | Score |
|---|:---:|
| **Exact Match (EM)** | **100.00%** |
| **Token F1 Score** | **100.00%** |
| **Token Precision** | **100.00%** |
| **Token Recall** | **100.00%** |

### Qualitative QA Samples:
1. **Q**: *"What is the primary architecture of the Transformer model based on?"*
   - **Predicted Answer**: `"attention mechanisms"` (Confidence: **0.9821**)
2. **Q**: *"What does BERT stand for?"*
   - **Predicted Answer**: `"Bidirectional Encoder Representations from Transformers"` (Confidence: **0.5448**)
3. **Q**: *"What procedure does BERT use to train a deep bidirectional representation?"*
   - **Predicted Answer**: `"Masked Language Model"` (Confidence: **0.4749**)

*Artifacts*:
- Result Metrics: [`experiments/results/qa/qa_results.json`](file:///c:/Pinak/Project/NLP/ResearchGPT/experiments/results/qa/qa_results.json)

---

## 9. Grounded RAG Pipeline (Task 17)

Evaluated the end-to-end Retrieval-Augmented Generation pipeline across indexed multi-document scientific corpora with provenance source citations:

- **Retrieval Engine**: `VectorRetrievalService` (384-dim Sentence-BERT Flat IP)
- **QA Reader**: `QuestionAnsweringService` (`distilbert-base-cased-distilled-squad`)
- **Anti-Hallucination Guard**: Configurable similarity & confidence thresholding

### End-to-End Grounded Generation Samples:
1. **Query**: *"How many layers are stacked in the Transformer encoder?"*
   - **Grounded Answer**: `"N = 6"` (Confidence: **0.7742**)
   - **Citation**: `vaswani_2017_transformer_chunk_2` (Similarity: **0.6919**)
   - **Latency**: Total: 2933 ms (Retrieval: 12.4 ms, QA: 2921 ms)
2. **Query**: *"How many total parameters does BERT base have?"*
   - **Grounded Answer**: `"110 million"` (Confidence: **0.7389**)
   - **Citation**: `devlin_2018_bert_chunk_2` (Similarity: **0.5365**)
   - **Latency**: Total: 126.8 ms (Retrieval: 10.2 ms, QA: 116.6 ms)
3. **Unanswerable Query**: *"What is the average rainfall in the Amazon rainforest?"*
   - **Grounded Answer**: `"I could not find sufficient relevant evidence in the indexed research papers to answer this question."`
   - **Status**: Grounded=False | Confidence=0.0000 | Citations=0

---

## 10. Full System Integration & Production API Layer (Task 18)

Comprehensive RESTful integration covering all 10 PRD functional requirements with dependency injection, singleton caching (`deps.py`), performance monitoring middleware, and end-to-end integration tests.

### API Router Coverage & Endpoints:

| Router | Method & Endpoint | Core Model / Service | Latency Header | Test Status |
|---|---|---|---|:---:|
| **Discovery & Health** | `GET /`, `GET /api/v1/health`, `GET /api/v1/health/detailed` | Subsystem Readiness Probes | `X-Process-Time-Ms` | **Passed** |
| **Documents** | `POST /api/v1/documents/upload`, `POST /api/v1/documents/{id}/extract` | `PdfReader` Page Extractor | `X-Process-Time-Ms` | **Passed** |
| **Preprocessing** | `POST /api/v1/preprocessing/classical`, `POST /api/v1/preprocessing/transformer` | NLTK Pipeline & RegEx Cleaner | `X-Process-Time-Ms` | **Passed** |
| **Classification** | `POST /api/v1/classification/predict`, `POST /api/v1/classification/batch` | `ClassicalClassifierService` / Transformer | `X-Process-Time-Ms` | **Passed** |
| **NER** | `POST /api/v1/ner/extract`, `POST /api/v1/ner/batch` | `BERTNERService` / `CRFNERService` | `X-Process-Time-Ms` | **Passed** |
| **Summarization** | `POST /api/v1/summarization/summarize` | `t5-small` Beam Search | `X-Process-Time-Ms` | **Passed** |
| **Similarity** | `POST /api/v1/similarity/compare`, `POST /api/v1/similarity/batch` | `Sentence-BERT` / TF-IDF | `X-Process-Time-Ms` | **Passed** |
| **Retrieval** | `POST /api/v1/retrieval/index-document`, `POST /api/v1/retrieval/search`, `GET /status` | `VectorRetrievalService` Flat IP | `X-Process-Time-Ms` | **Passed** |
| **Question Answering** | `POST /api/v1/qa/answer`, `POST /api/v1/qa/batch` | `distilbert-base-cased-distilled-squad` | `X-Process-Time-Ms` | **Passed** |
| **Grounded RAG** | `POST /api/v1/rag/query` | RAG Pipeline Orchestrator | `X-Process-Time-Ms` | **Passed** |
| **Integrated Analysis** | `POST /api/v1/analysis/paper` | Multi-Modal Pipeline (NER + Summary + Index) | `X-Process-Time-Ms` | **Passed** |

### Middleware & Cross-Cutting Capabilities:
- **CORS**: Fully configured for modern browser interfaces.
- **Request Tracing**: `X-Request-ID` generated or propagated across all HTTP lifecycle handlers.
- **Performance Profiling**: Sub-millisecond `X-Process-Time-Ms` tracking header on every response.
- **Exception Sanitization**: Centralized 500 error handler with stack-trace preservation in backend logs.
- **Integration Test Suite**: [`tests/integration/test_api_endpoints.py`](file:///c:/Pinak/Project/NLP/ResearchGPT/tests/integration/test_api_endpoints.py) and [`tests/integration/test_end_to_end_workflow.py`](file:///c:/Pinak/Project/NLP/ResearchGPT/tests/integration/test_end_to_end_workflow.py) verifying full multi-step research paper ingestion, indexing, and synthesis.

---

## 11. Frontend User Interface & Interactive Workspace (Task 19)

Complete modern glassmorphic web application built with semantic HTML5, Vanilla CSS3, and ES6+ modules. Directly served from the FastAPI backend with turnkey static asset routing.

### Screen & Interface Suite:

| Screen Identifier | Viewport / Section Name | Core Interactive Capabilities | Status |
|---|---|---|:---:|
| `#screen-dashboard` | **System Overview & Health** | Live subsystem readiness probes, quick launchers, key performance metrics | **Verified** |
| `#screen-documents` | **PDF Ingestion & Extraction** | Drag-and-drop PDF upload, page-aware text stream browser, preprocessor inspector | **Verified** |
| `#screen-rag` | **Grounded RAG Assistant** | Conversational chat interface, top-K chunk selector, similarity slider, citations drawer | **Verified** |
| `#screen-analysis` | **Paper Deep-Dive Analysis** | Single-pass executive summary, color-coded entity tagger (`PER`, `ORG`, `LOC`, `MISC`), auto-indexing | **Verified** |
| `#screen-playground` | **Multi-Task NLP Playground** | 5 interactive tabs: Classification, NER, Summarization, Semantic Similarity, Extractive QA | **Verified** |
| `#screen-benchmarks` | **Benchmark Leaderboard** | Full comparative metrics tables and test splits across all 5 benchmark datasets | **Verified** |

### Static Serving & Content Negotiation:
- **Static Assets**: Accessible at `/ui/`, `/css/style.css`, and `/js/app.js`.
- **Content Negotiation**: Browser visits to `http://localhost:8000/` automatically serve `index.html`, while API clients receive the JSON service discovery payload.
- **Integration Test Suite**: [`tests/integration/test_frontend_static_serving.py`](file:///c:/Pinak/Project/NLP/ResearchGPT/tests/integration/test_frontend_static_serving.py) passing with 4/4 assertions.

