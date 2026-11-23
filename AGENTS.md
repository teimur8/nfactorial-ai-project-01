# Agent Instructions for Kinomaniac Project

This document provides guidelines and architectural context for autonomous AI agents collaborating on the Kinomaniac codebase.

## 1. Architecture & Core Invariants
- **Grounding & Guardrails:** Never trust LLM fact retrieval. All facts (title, year, duration, genres) are strictly sourced from the local catalog object.
- **Contract Integrity:** The `pick(client, model, query, films)` function must:
  - Return `(outcome, ids)` where `ids` is empty unless `outcome == "ok"`.
  - Print recommended movie details directly to stdout from the passed `films` catalog.
  - Never read `films.json` from disk inside `pick` (must operate strictly on the passed `films` list).
  - Log every invocation to `calls.jsonl`.
- **Tier Invariants:**
  - **Common:** Pure catalog grounding, ID validation, duplicate filtering, max 3 limit, error handling.
  - **Rare:** Deterministic filter extraction with `Filters` Pydantic model (`step="filters"`), filtering catalog before final choice prompt (`step="choice"`).
  - **Epic:** Algorithmic non-LLM candidate retrieval via `candidates(query, films, k=30)` under 20s; `pick_big` sends at most 30 candidates into the LLM prompt.
  - **Legendary:** Support `status="clarify"`, strict CI/CD with pytest, EARS specifications, pre-commit configuration.

## 2. Code Quality & Standards
- **Python Version:** Python 3.12 managed via `uv`.
- **Formatting & Linting:** Strict adherence to PEP 8, verified via `ruff` and `mypy`.
- **Testing:** All core logic and edge cases must have unit tests with mocks in `tests/test_kino.py`. No real API calls are allowed in tests.
- **Language Policy:** Keep all code comments and documentation in English.
