# Preprocessing

## 1. Purpose

Preprocessing converts raw paper text and benchmark data into representations suitable for classical NLP, neural models, transformers, and semantic retrieval.

## 2. Classical NLP Pipeline

```text
Raw Text
   ↓
Sentence Segmentation
   ↓
Tokenization
   ↓
Lowercasing
   ↓
Stopword Removal
   ↓
Lemmatization
   ↓
TF-IDF / Classical Model
```

## 3. Transformer Pipeline

Transformers should avoid unnecessarily destructive preprocessing.

```text
Raw / lightly cleaned text
        ↓
Tokenizer
        ↓
Token IDs + attention mask
        ↓
Transformer
```

Do not remove words solely because they are stopwords when doing so could alter model semantics.

## 4. Research-Paper Text

PDF extraction may contain:

- headers;
- footers;
- page numbers;
- broken line wraps;
- repeated section titles;
- references;
- tables;
- equations.

Cleaning rules must be conservative.

The system should preserve source text needed for retrieval and citation.

## 5. Chunking

Chunking is separate from generic preprocessing.

A chunk should retain:

- document ID;
- page range;
- source text;
- chunk ID.

Chunking strategy must be documented when implemented.

## 6. Reproducibility

Every preprocessing experiment should record:

- normalization;
- tokenization;
- stopword policy;
- lemmatization/stemming policy;
- truncation;
- maximum length;
- chunk size;
- overlap;
- tokenizer version where relevant.

## 7. Testing

Tests should cover:

- empty text;
- whitespace;
- punctuation;
- Unicode;
- very long text;
- repeated headers/footers where supported;
- page boundaries;
- preservation of source metadata.
