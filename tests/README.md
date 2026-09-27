# ResearchGPT Tests

Tests are organized by scope.

```text
tests/
├── unit/
├── integration/
└── system/
```

## Unit Tests

Test isolated functions and services.

## Integration Tests

Test component boundaries such as:

- API → document service;
- API → NLP service;
- embedding → FAISS;
- retrieval → QA.

## System Tests

Test complete flows such as:

```text
Upload PDF
→ Extract
→ Analyze
→ Index
→ Ask
→ Retrieve
→ Answer
```

No system-level test should depend on an external paid service unless explicitly configured for that test.
