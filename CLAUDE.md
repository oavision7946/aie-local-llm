# aie-local-llm

Serve open-weight models locally on an NVIDIA DGX Spark (aarch64, unified GPU/CPU memory) and expose them through a single OpenAI-compatible endpoint, with an optional RAG layer. Inference engines (vLLM, SGLang, LMCache-accelerated vLLM) each run in their own container. The FastAPI gateway in `app/` is only an HTTP client to the active engine.

> **This file is public.** Keep it free of roadmap, customer, employer, and internal design details.

## Tech stack

- Python 3.13, with dependencies and environments managed by **uv** (`uv.lock` is committed)
- FastAPI gateway (`app/`), Streamlit test UI (`ui/`)
- Engines in Docker: `docker/<engine>/run.sh` scripts and `docker/docker-compose.yml` profiles
- Pydantic models for configuration and for the API's request and response shapes
- Configuration in YAML under `config/` (`engines.yaml`, `rag.yaml`). Never hard-code a model, engine URL, or port in code.

## Commands

```bash
uv sync                                   # install dependencies
uv run pytest tests/unit                  # unit tests
uv run pytest tests/contract              # API contract tests (separate suite)
uv run ruff check . && uv run ruff format --check .   # lint and format check
uv run uvicorn app.main:app --port <port>             # gateway (binds 127.0.0.1 by default)
uv run streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8502
bash docker/vllm/run.sh                   # or docker/sglang/run.sh, docker/vllm-lmcache/run.sh
bash docker/stop.sh                       # stop the running engine
```

Contract (API) tests live apart from unit tests and are run separately. Tests must not need a GPU, a running engine, or network access; fake the engine client.

## Conventions

- Every change ships with tests, passes lint, and updates docs (`README.md`, `docs/`) when behavior or commands change.
- Validate all API inputs and outputs with Pydantic models; keep the gateway OpenAI-compatible (extra fields such as `rag` must stay optional).
- Configuration lives in YAML in `config/`; ports and memory settings are configurable through environment variables or config, never fixed in code.
- Secrets (for example `HF_TOKEN`) live in `.env` files or the shell environment and are never committed, printed, or written to other files. Keep a `.env.example` with placeholder values only.
- Keep changes small and focused. Run tests and linters before committing.

## Running engines safely

This host can be on a shared network, so local services must not be reachable from it.

- **Publish ports on loopback only.** Write every port mapping with an explicit host address: `127.0.0.1:${PORT}:8000`, never `${PORT}:8000`. Docker's daemon-wide default bind address applies only to its default bridge network. Docker Compose creates its own network, so a mapping without a host address publishes on all interfaces.
- After starting any container, check `docker ps --format '{{.Names}} {{.Ports}}'`. If anything shows `0.0.0.0` or `[::]`, stop it and fix the mapping. A server listening on `0.0.0.0` *inside* the container is fine when the host side is bound to `127.0.0.1`.
- Do not use `--host 0.0.0.0` for the gateway, and run Streamlit with `--server.address 127.0.0.1`.
- **Ports in use on this host:** other local services here already use 8123, 9000 and 54322, so don't run the gateway on 9000. Check `ss -ltn` before choosing a port.
- **Pin images** to an exact tag or digest (no `latest`), so benchmarks are reproducible.
- **Memory is shared.** The GPU and CPU share one memory pool, and other services run on the same machine. Set `--gpu-memory-utilization` (or the engine's equivalent) deliberately, and record the value with any measurement.
- Model downloads are large. Check free disk space first, and reuse the shared Hugging Face cache (`~/.cache/huggingface`).
- Stop the engines you started when you're done.

## Public-repo hygiene

This repository is public. Treat everything you write into it as published.

- Do not put any of the following in tracked files (code, comments, docstrings, docs, README, test fixtures, commit messages, PR or issue text): roadmap or unreleased plans, customer names or data, employer names, internal design rationale, or details of other private projects that use this service.
- Test data and benchmark prompts must be synthetic. Never commit real logs, traces, or documents from a customer or employer.
- `.internal/`, `CLAUDE.local.md` and `ProjectPrompts.md` are gitignored. Never stage them, never use `git add -f` on them, and never remove their ignore rules.
- Write comments and docs in generic technical terms: what the code does and how to use it.
- If you need planning context and `CLAUDE.local.md` is not present, ask the user rather than guessing.

## Git workflow

- Do not commit or push without the user's review. Leave changes uncommitted, run the checks, and summarize the diff.
- Work on `main` unless the user names a branch. The user pushes.
