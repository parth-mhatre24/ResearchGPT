# ResearchGPT Development Guide

## 1. Development Principle

ResearchGPT is implemented one controlled task at a time.

The objective is:

> Correctness first, completeness second, speed third.

## 2. Source of Truth

The GitHub repository is the source of truth for:

- requirements;
- architecture;
- workflow;
- implementation;
- experiments;
- tests;
- project status.

## 3. Antigravity Rules

Before changing code, Antigravity must read:

1. `README.md`
2. `PRD.md`
3. `PROJECT_STATUS.md`
4. relevant documentation in `docs/`
5. the current task requirements.

## 4. One Task at a Time

Antigravity must not implement future tasks.

If Task 04 is active, it must not silently implement:

- classification;
- NER;
- RAG;
- frontend;
- unrelated refactoring.

## 5. Required Task Cycle

```text
1. Understand requirements
2. Inspect existing code
3. Identify affected files
4. Implement current task
5. Add/update tests
6. Run tests
7. Review implementation
8. Fix issues
9. Run tests again
10. Update documentation
11. Update PROJECT_STATUS.md
12. Report results
13. Stop
14. Wait for human approval
```

## 6. Completion Report

Every task report must include:

- task name;
- requirements addressed;
- files created;
- files modified;
- tests added;
- tests executed;
- test results;
- manual verification;
- issues discovered;
- issues fixed;
- remaining limitations;
- security considerations;
- performance considerations;
- technical debt;
- next task recommendation.

## 7. Human Approval

Antigravity must never self-approve a task.

Only the project owner can move the project to the next task.

## 8. Testing

Every meaningful implementation should have an appropriate test.

Prefer:

- unit tests for pure logic;
- integration tests for component boundaries;
- end-to-end tests for critical user flows.

## 9. ML Experiment Rules

Record:

- dataset;
- version;
- split;
- seed;
- model;
- tokenizer;
- hyperparameters;
- training settings;
- metrics;
- environment.

Never invent metrics.

## 10. Data Leakage

Before reporting a benchmark result, verify that evaluation data was not used for:

- fitting;
- vocabulary construction;
- hyperparameter tuning;
- threshold tuning;
- model selection.

## 11. Git Workflow

Recommended:

```text
main
  ↑
task/01-repository-audit
task/02-project-foundation
task/03-pdf-upload
...
```

Keep `main` limited to reviewed and approved work.

Use focused commits.

## 12. Security

Never commit:

- API keys;
- passwords;
- tokens;
- private documents;
- `.env` files;
- unnecessary user data.

Uploaded PDFs must be treated as untrusted input.

## 13. Architecture Changes

Do not silently change a documented primary architecture.

If a change is necessary:

1. document the reason;
2. identify affected components;
3. update the architecture documentation;
4. obtain approval when the change is material.

## 14. Review Prompt

Use this review prompt after implementation:

```text
TASK REVIEW — DO NOT MODIFY CODE

Review only the current task.

Check:
- requirements;
- implementation;
- tests;
- integration;
- security;
- performance;
- maintainability;
- documentation.

Report:
- what is correct;
- what is missing;
- bugs;
- risks;
- technical debt;
- exact required fixes.

Do not modify code.
Do not begin the next task.
```

## 15. Approval Prompt

If approved:

```text
Task reviewed and approved.
Proceed to the next task only.
Read the next task requirements before modifying code.
```

If not approved:

```text
Task is not approved.

Fix only these issues:
[list issues]

Run the relevant tests again.
Report the changes and test results.
Do not start the next task.
```
