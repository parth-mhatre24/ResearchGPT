# Evaluation

## 1. Evaluation Principles

Every experiment must define:

- task;
- dataset;
- split;
- baseline;
- model;
- configuration;
- metric;
- random seed;
- environment.

No metric may be fabricated or copied from an unrelated experiment.

## 2. Classification

Report:

- Accuracy;
- Precision;
- Recall;
- F1;
- Confusion Matrix.

For multi-class experiments, document the averaging strategy:

- macro;
- micro;
- weighted.

## 3. NER

Report entity-level:

- Precision;
- Recall;
- F1.

Token-level accuracy alone is insufficient as the primary NER metric.

## 4. Summarization

Report:

- ROUGE-1;
- ROUGE-2;
- ROUGE-L.

BERTScore may be reported as an additional metric.

The evaluation must specify:

- reference summaries;
- generated summaries;
- tokenization/normalization settings;
- aggregation method.

## 5. Semantic Similarity

For STS-B-style experiments, report appropriate correlation metrics such as:

- Pearson;
- Spearman.

MSE may be reported where appropriate.

## 6. QA

Report:

- Exact Match;
- Token F1;
- Precision;
- Recall.

The normalization rules for Exact Match must be documented.

## 7. Retrieval

When retrieval evaluation is introduced, document:

- query set;
- relevant-document/chunk definition;
- top-k;
- retrieval metric;
- index type.

Potential metrics include Recall@k, Precision@k, MRR, or another justified metric.

## 8. End-to-End Evaluation

End-to-end testing should verify:

```text
Upload
→ extraction
→ analysis
→ indexing
→ retrieval
→ QA/RAG
→ source display
```

## 9. Experiment Record

Recommended result format:

```text
Experiment ID:
Date:
Task:
Dataset:
Dataset version:
Split:
Model:
Checkpoint:
Tokenizer:
Preprocessing:
Seed:
Hyperparameters:
Metric:
Result:
Environment:
Notes:
```

## 10. Comparison Rule

Model comparisons are valid only when the evaluation conditions are sufficiently comparable.

A higher number is not automatically evidence of a better system if:

- datasets differ;
- splits differ;
- preprocessing differs materially;
- metrics differ;
- leakage exists.
