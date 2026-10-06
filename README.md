# AI Engineering - Running Local Models on NVIDIA DGX Spark

A local-model question-answering and RAG gateway for this DGX Spark. It runs inference via **vLLM**,
**SGLang**, or **LMCache**-accelerated vLLM (each in its own Docker container, exposing an
OpenAI-compatible API), adds a retrieval-augmented-generation layer on top, and re-exposes everything as
a single OpenAI-compatible `/v1/chat/completions` endpoint — so it can be used as a drop-in local LLM
provider for any application that already speaks the OpenAI API.

New to this box? Start with the tutorials:

- [Tutorial 1: Run a local LLM](docs/tutorial-1-run-a-local-llm.md) — serve `Qwen/Qwen3.5-9B` through
  vLLM and talk to it via the gateway. No RAG, shortest path to a working endpoint.
- [Tutorial 2: Run an LLM + an embedding model](docs/tutorial-2-llm-and-embeddings.md) — add
  `nvidia/Nemotron-3-Embed-8B-BF16` alongside `Qwen/Qwen3.8-27B` so RAG retrieval uses a real
  retrieval-tuned embedder, including how to budget GPU memory across both models.

## Why OpenAI-compatible?

Most LLM client libraries and frameworks support pointing at a custom `base_url` for an
"OpenAI-compatible" provider. Pointing that setting at this service's `/v1` makes it a local-model
backend for that application with **no code changes on the application side**, e.g.:

```python
from openai import OpenAI

client = OpenAI(base_url="http://<this-host>:9000/v1", api_key="not-needed")
```

RAG is opted into per-request via an extra `rag` field in the request body (ignored by plain OpenAI
clients that don't send it):

```json
{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "messages": [{"role": "user", "content": "..."}],
  "rag": {"enabled": true, "corpus": "docs", "top_k": 5}
}
```

## Architecture

- **Engines** (`docker/`): vLLM, SGLang, and LMCache-accelerated vLLM each run as their own Docker
  container. This app never runs GPU inference in-process — it's purely an HTTP client to whichever
  engine is configured as `active` in `config/engines.yaml`.
- **Gateway** (`app/`): FastAPI app. `app/services/engines/` talks to the active engine over the OpenAI
  chat completions API. `app/services/rag/` chunks and embeds ingested documents (CPU, via
  sentence-transformers) into an in-memory, namespaced-by-corpus vector store, and retrieves context for
  RAG-enabled requests. `app/services/chat/` ties the two together.
- **UI** (`ui/`): a small Streamlit app for manually testing QA and RAG.

See `config/engines.yaml` and `config/rag.yaml` for all tunables.

## Running an engine

Set `HF_TOKEN` first so the engine's Hugging Face downloads are authenticated (higher rate limits,
faster downloads, required for gated models):

```bash
export HF_TOKEN=hf_xxx  # https://huggingface.co/docs/hub/en/security-tokens
```

Each script pulls/launches its container and prints a readiness check. They default to the already-cached
`Qwen/Qwen2.5-3B-Instruct` model; override with `VLLM_MODEL`/`SGLANG_MODEL` env vars.

```bash
bash docker/vllm/run.sh           # port 8000
bash docker/vllm-lmcache/run.sh   # port 8001, uses the local lmcache-vllm:spark image
bash docker/sglang/run.sh         # port 30000, pulls lmsysorg/sglang if not present

bash docker/stop.sh               # stop whichever is running
```

Or via Compose profiles:

```bash
cd docker && docker compose --profile vllm up
```

Only one engine needs to be running at a time; set `active` in `config/engines.yaml` to match, or edit
`base_url`/`model`/`enabled` for any entry.

> **Note on this hardware:** DGX Spark's unified memory means the engine and the embedding model compete
> for the same GPU memory pool. `config/rag.yaml`'s `embeddings.device` defaults to `cpu` for this reason
> — flip it to `cuda` only if you've left headroom in the engine's `--gpu-memory-utilization`.

## Running the gateway

```bash
uv sync
uv run uvicorn app.main:app --port 9000
```

```bash
curl http://localhost:9000/health
curl http://localhost:9000/v1/models
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-3B-Instruct", "messages": [{"role": "user", "content": "hi"}]}'
```

Manage documents for RAG:

```bash
curl -X POST http://localhost:9000/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "...", "corpus": "docs"}'

curl http://localhost:9000/documents?corpus=docs
curl -X DELETE http://localhost:9000/documents/<doc_id>
```

## Running the UI

```bash
uv sync --extra ui
uv run streamlit run ui/app.py --server.port 8502
```

Open `http://localhost:8502`. Point it at a different API with `API_BASE_URL=http://host:port`.

## Running the tests

```bash
uv run pytest
```

No GPU, Docker, or network access required — engines and embeddings are replaced with deterministic
fakes (`app/services/engines/fake.py`, `app/services/rag/embeddings/fake.py`) via FastAPI's
`dependency_overrides` in `tests/conftest.py`.
