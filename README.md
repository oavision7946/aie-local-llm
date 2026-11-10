# AI Engineering - Running Local Models on NVIDIA DGX Spark

A local-model question-answering and RAG gateway for this DGX Spark. It runs inference via **vLLM**,
**SGLang**, **LMCache**-accelerated vLLM, or the **llama.cpp** server (each in its own Docker container, exposing an
OpenAI-compatible API), adds a retrieval-augmented-generation layer on top, and re-exposes everything as
a single OpenAI-compatible `/v1/chat/completions` endpoint — so it can be used as a drop-in local LLM
provider for any application that already speaks the OpenAI API.

New to this box? Start with the tutorials:

- [Tutorial 1: Run a local LLM](docs/tutorial-1-run-a-local-llm.md) — serve `Qwen/Qwen3.5-9B` through
  vLLM and talk to it via the gateway. No RAG, shortest path to a working endpoint.
- [Tutorial 2: Run an LLM + an embedding model](docs/tutorial-2-llm-and-embeddings.md) — add
  `nvidia/Nemotron-3-Embed-8B-BF16` alongside `Qwen/Qwen3.8-27B` so RAG retrieval uses a real
  retrieval-tuned embedder, including how to budget GPU memory across both models.
- [Tutorial 3: Configure an embedding model](docs/tutorial-3-embedding-models.md) — verified configs for
  Qwen3-Embedding, BGE-M3, EmbeddingGemma and others, with prompt-prefix and `trust_remote_code` handling.

## Why OpenAI-compatible?

Most LLM client libraries and frameworks support pointing at a custom `base_url` for an
"OpenAI-compatible" provider. Pointing that setting at this service's `/v1` makes it a local-model
backend for that application with **no code changes on the application side**, e.g.:

```python
from openai import OpenAI

client = OpenAI(base_url="http://<this-host>:9100/v1", api_key="not-needed")
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

- **Engines** (`docker/`): vLLM, SGLang, LMCache-accelerated vLLM and llama.cpp each run as their own Docker
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

Each script pulls/launches its container and prints a readiness check. vLLM, vLLM+LMCache and SGLang
default to the already-cached `Qwen/Qwen2.5-3B-Instruct` model; override with `VLLM_MODEL`/`SGLANG_MODEL`.
Each example below runs the engine, polls its own stated health check until it answers, confirms the
port is loopback-only, then stops it — copy-paste one at a time (`docker/stop.sh` is safe to call even if
nothing is running, so you don't need to track what's up before moving to the next engine).

**vLLM** (`127.0.0.1:8000`):

```bash
bash docker/vllm/run.sh
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health | grep -q 200; do sleep 3; done
docker ps --format '{{.Names}} {{.Ports}}'   # confirm 127.0.0.1:8000, not 0.0.0.0
curl http://127.0.0.1:8000/v1/models
bash docker/stop.sh
```

**vLLM + LMCache** (`127.0.0.1:8001`) — same image, with the LMCache KV-cache connector enabled:

```bash
bash docker/vllm-lmcache/run.sh
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health | grep -q 200; do sleep 3; done
docker ps --format '{{.Names}} {{.Ports}}'   # confirm 127.0.0.1:8001, not 0.0.0.0
curl http://127.0.0.1:8001/v1/models
bash docker/stop.sh
```

**SGLang** (`127.0.0.1:30000`):

```bash
bash docker/sglang/run.sh
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health | grep -q 200; do sleep 3; done
docker ps --format '{{.Names}} {{.Ports}}'   # confirm 127.0.0.1:30000, not 0.0.0.0
curl http://127.0.0.1:30000/v1/models
bash docker/stop.sh
```

**llama.cpp** (`127.0.0.1:8002`) — serves a GGUF file instead of a Hugging Face model id, so download one
first (a small model keeps this quick; swap in whatever `LLAMACPP_HF_REPO`/`LLAMACPP_HF_FILE` you actually
want to run):

```bash
uvx --from 'huggingface_hub[cli]' hf download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf

LLAMACPP_HF_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF LLAMACPP_HF_FILE=qwen2.5-3b-instruct-q4_k_m.gguf \
  bash docker/llamacpp/run.sh
until curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8002/health | grep -q 200; do sleep 3; done
docker ps --format '{{.Names}} {{.Ports}}'   # confirm 127.0.0.1:8002, not 0.0.0.0
curl http://127.0.0.1:8002/v1/models
bash docker/stop.sh
```

Or via Compose profiles:

```bash
cd docker && docker compose --profile vllm up
```

Only one engine needs to be running at a time; set `active` in `config/engines.yaml` to match, or edit
`base_url`/`model`/`enabled` for any entry.

Every image is pinned to an exact tag and digest in its script (`VLLM_IMAGE`, `SGLANG_IMAGE`,
`LLAMACPP_IMAGE`, ...), so measurements are reproducible; override the variable to try another build.
Ports, context length and memory settings are environment variables too; see the top of each script.

### Memory settings

DGX Spark's CPU and GPU share one memory pool, so `--gpu-memory-utilization` (vLLM) and
`--mem-fraction-static` (SGLang) are fractions of the **whole machine's** memory, not of a separate GPU.
The scripts default to `0.4`, which leaves room for other services; raise it only when nothing else needs
the memory. llama.cpp allocates its KV cache up front from `LLAMACPP_CTX_SIZE`. Note that
`docker stats` does not show memory a container allocates on the GPU; use `free -m` to see the real total.

The same applies to embedding models: the engine and the embedder compete for the same pool.
`config/rag.yaml`'s `embeddings.device` defaults to `cpu` for this reason. Flip it to `cuda` only if you've
left headroom in the engine's memory fraction.

### Running engines safely

- Every script and Compose profile publishes its port on `127.0.0.1` only. Docker's daemon-wide default
  bind address does not apply to the networks Compose creates, so a mapping without a host address would
  publish on all interfaces. After starting anything, check
  `docker ps --format '{{.Names}} {{.Ports}}'` for `0.0.0.0` or `[::]`.
- `HF_TOKEN` is passed to containers by name (`-e HF_TOKEN`), so its value does not appear in the
  `docker run` command line.
- Engines run as your user (`--user $(id -u):$(id -g)`), not root, so files they write to the shared
  Hugging Face cache stay owned by you. The cache is mounted at `/hf-cache` (`HF_HOME`), and `HOME=/tmp`
  gives compile and JIT caches a writable place. With Compose, pass your IDs:
  `UID=$(id -u) GID=$(id -g) docker compose --profile vllm up` (the default is 1000:1000).
- If earlier root-run containers left root-owned files in the cache, engines log harmless "Permission
  denied ... tree cache" warnings; fix ownership once with
  `sudo chown -R "$(id -u):$(id -g)" ~/.cache/huggingface`.

## Running the gateway

```bash
uv sync
uv run uvicorn app.main:app --port 9100
```

Any free port works (check with `ss -ltn`); the examples use 9100 because 9000 is often taken by other
local services (for example, ClickHouse's native protocol). uvicorn binds `127.0.0.1` by default.

```bash
curl http://127.0.0.1:9100/health
curl http://127.0.0.1:9100/v1/models
curl -X POST http://127.0.0.1:9100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-3B-Instruct", "messages": [{"role": "user", "content": "hi"}]}'
```

Manage documents for RAG:

```bash
curl -X POST http://127.0.0.1:9100/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "...", "corpus": "docs"}'

curl http://127.0.0.1:9100/documents?corpus=docs
curl -X DELETE http://127.0.0.1:9100/documents/<doc_id>
```

## Running the UI

```bash
uv sync --extra ui
uv run streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8502
```

Open `http://127.0.0.1:8502`. The UI talks to `http://127.0.0.1:9100` by default; point it at a different
API with `API_BASE_URL=http://host:port`.

## Benchmarking an engine

`bench/serve_bench.py` measures any OpenAI-compatible engine: load time, decode tokens/s (one stream and
concurrent streams), prefill throughput and time to first token for long prompts, tool calling, and
JSON-schema structured output. Prompts are synthetic and seeded. See
[docs/benchmarking.md](docs/benchmarking.md) for the workflow and the latest results on this machine.

```bash
uv run python bench/serve_bench.py run --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-3B-Instruct --label vllm-3b
```

## Running the tests

```bash
uv run pytest
```

No GPU, Docker, or network access required — engines and embeddings are replaced with deterministic
fakes (`app/services/engines/fake.py`, `app/services/rag/embeddings/fake.py`) via FastAPI's
`dependency_overrides` in `tests/conftest.py`.
