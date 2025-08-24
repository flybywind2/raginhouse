# Repository Guidelines

## Project Structure & Module Organization
- `src/agents/`: LangGraph workflows (`rag_workflow.py`, `optimized_rag_workflow.py`).
- `src/api/`: FastAPI routes (`routes.py`).
- `src/clients/`: External service clients (LLM, retriever, documents).
- `src/services/`: Core services (RRF/MMR, caching, document processing, feedback).
- `src/core/`: App configuration (`config.py`).
- `templates/`, `static/`: Web UI assets. Root: `main.py`, `run_dev.py`, tests (`test_*.py`).

## Build, Test, and Development Commands
- Install: `pip install -r requirements.txt`
- Run dev server: `python run_dev.py` (UI at `http://localhost:8080`, docs at `/docs`).
- Alternate run: `uvicorn main:app --reload --host 0.0.0.0 --port 8080`
- Tests (pytest): `pytest -q`
- Test scripts (direct): `python test_workflow_structure.py`
- Lint: `flake8 src tests .`
- Format: `black src tests .`
- Type check: `mypy src`

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, UTF-8.
- Filenames: `snake_case.py`. Modules and packages mirror domain (agents, services, api).
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Imports: absolute within `src` (e.g., `from src.services.rag_service import RAGService`).
- Use `black`, `flake8`, and `mypy` before pushing. Keep functions small and pure where possible.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio` for async workflows.
- Locations: root `test_*.py` (current pattern). Prefer new tests under `tests/` if added.
- Naming: files `test_*.py`, tests `test_*` functions. Use fakes/mocks for external calls.
- Run: `pytest -q` or per-file `pytest test_workflow_structure.py -q`.
- Aim for coverage on agents, services, and API routes; validate graph nodes/edges and error paths.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (e.g., `feat: add RRF fusion`, `fix: handle cache TTL`).
- PRs: include purpose, linked issues, screenshots of UI changes, and test instructions.
- Checks: passing tests, lint, type checks; update docs (`README.md`, this guide) when relevant.

## Security & Configuration Tips
- Configure via `.env` (copy from `.env.example`). Do not commit secrets.
- Keys: `OPENAI_API_KEY`, `RAG_API_KEY`, `DEP_TICKET`, `REDIS_URL`, etc. See `src/core/config.py`.
- Avoid logging sensitive tokens; prefer dependency injection for clients; validate upload types in `/ingest/*`.

## Architecture Overview
- Request flow: FastAPI (`src/api/routes.py`) → workflow (`src/agents/*`) → services/clients → response with citations and latency.
- Retrieval: BM25/kNN/CC with RRF fusion; context via MMR; optional rerank; async critique/refine steps.
