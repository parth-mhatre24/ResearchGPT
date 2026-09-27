# ResearchGPT Backend

The backend provides the application API and orchestrates:

- PDF processing;
- NLP services;
- embeddings;
- retrieval;
- QA/RAG.

## Planned Structure

```text
backend/
└── app/
    ├── api/
    ├── core/
    ├── models/
    ├── services/
    └── main.py
```

## Development Rule

Backend implementation must follow the task currently recorded in `PROJECT_STATUS.md`.

Do not add model services before their corresponding task is approved.

## Environment

The backend environment and dependency versions are established during the project-foundation task.
