# Data Directory

This directory is reserved for local datasets and document-processing data.

## Policy

Do not commit large datasets or user-uploaded research papers to Git.

Use this directory for local development only unless a small, redistributable fixture is explicitly approved.

## Dataset Sources

See [`../docs/Dataset.md`](../docs/Dataset.md).

## Recommended Local Layout

```text
data/
├── raw/
├── processed/
├── fixtures/
└── README.md
```

## Reproducibility

Every dataset used in an experiment must have:

- source;
- version;
- split;
- license;
- preprocessing record;
- download instructions.

## User Documents

Uploaded research papers should be handled by the application storage layer and should not be committed to Git.
