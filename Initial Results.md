
3. **CLI Training & Experiment Runner**:
   - Created [`scripts/train_classification.py`](file:///c:/Pinak/Project/NLP/ResearchGPT/scripts/train_classification.py) supporting configurable flags (`--dataset [sms_spam|imdb|sst2|all]`, `--model [naive_bayes|logistic_regression|svm|all]`, `--max-features`, `--ngram-min`, `--ngram-max`).
   - Automatically handles labeled vs. unlabeled splits (e.g., routing to the GLUE `validation` split for SST-2 benchmark evaluation).
   - Generates reproducible JSON result logs under [`experiments/results/classification/`](file:///c:/Pinak/Project/NLP/ResearchGPT/experiments/results/classification/).

4. **Testing & Quality Assurance**:
   - Created [`tests/unit/test_classical_classifier.py`](file:///c:/Pinak/Project/NLP/ResearchGPT/tests/unit/test_classical_classifier.py) with 10 unit tests covering pipeline creation, training, probabilistic inference, metric calculations, and persistence.
   - All **30 unit tests across the entire repository pass with zero errors**.

---

### Empirical Benchmark Results

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
- On long document text (`imdb`), **Logistic Regression** led with **88.84% accuracy / F1**, directly replicating Paper 1's classical baseline target (~88%).
- On sentence-level syntax (`sst2`), **Logistic Regression** achieved **79.59% accuracy**, setting the baseline for comparison with BERT/DistilBERT in Task 08.

