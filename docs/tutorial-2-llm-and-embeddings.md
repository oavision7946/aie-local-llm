# Tutorial 2: Run an LLM + an embedding model together (Qwen3.8-27B + Nemotron-3-Embed-8B)

This walks through running a larger local LLM — [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
— **alongside** a dedicated embedding model —
[`nvidia/Nemotron-3-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16) — so RAG
retrieval uses a real retrieval-tuned embedder instead of the small default
(`BAAI/bge-small-en-v1.5`). Read [Tutorial 1](tutorial-1-run-a-local-llm.md) first if you haven't served a
local model through this gateway before.

## 0. Size the memory budget first

Both models are large enough that memory planning matters, even with this DGX Spark's 128GB of unified
memory shared between CPU and GPU:

| Model | Checkpoint | Approx. size |
|---|---|---|
| `Qwen/Qwen3.8-27B` | BF16 | ~52 GB |
| `nvidia/Qwen3.8-27B-NVFP4` | NVFP4 | ~15-18 GB |
| `nvidia/Nemotron-3-Embed-8B-BF16` | BF16 | ~16 GB |

Running the LLM in BF16 (~52 GB) plus vLLM's KV cache pool leaves little headroom for an 8B embedding
model on the same GPU. **Use the NVFP4 checkpoint for the LLM in this tutorial** — it's the same model,
quantized, and frees up enough memory to run the embedder on GPU too. If you'd rather not re-download a
different checkpoint, keep the embedder on CPU instead (see step 3) — that always works regardless of
how much GPU memory the LLM reserves.

## 1. Launch vLLM with the quantized LLM, leaving GPU headroom

Both checkpoints are several GB+; set `HF_TOKEN` first for authenticated (faster, higher rate limit)
downloads — see [Tutorial 1](tutorial-1-run-a-local-llm.md#0-prerequisites):

```bash
export HF_TOKEN=hf_xxx  # https://huggingface.co/docs/hub/en/security-tokens
```

```bash
cd aie-local-llm
VLLM_MODEL=nvidia/Qwen3.8-27B-NVFP4 \
VLLM_GPU_MEMORY_UTILIZATION=0.55 \
VLLM_EXTRA_ARGS="--quantization modelopt_fp4" \
  bash docker/vllm/run.sh
```

`VLLM_GPU_MEMORY_UTILIZATION=0.55` caps vLLM's reserved pool well under full GPU memory, leaving room for
the embedding model loaded in step 3. `--quantization modelopt_fp4` tells vLLM how to interpret the
NVFP4 weights (vLLM can usually auto-detect this from the checkpoint's `config.json`, but passing it
explicitly avoids relying on that).

Wait for readiness:

```bash
until curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health | grep -q 200; do
  sleep 3
done
echo "vLLM is ready"
```

Check how much GPU memory is actually in use before moving on:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 2. Point the gateway at the LLM

```yaml
# config/engines.yaml
active: vllm

engines:
  - name: vllm
    base_url: http://127.0.0.1:8000/v1
    model: nvidia/Qwen3.8-27B-NVFP4
    enabled: true
```

## 3. Configure the embedding model

```yaml
# config/rag.yaml
embeddings:
  model: nvidia/Nemotron-3-Embed-8B-BF16
  device: cuda   # headroom permitting, per step 0 — use "cpu" if you skipped the NVFP4 swap

chunking:
  chunk_size: 500
  chunk_overlap: 50

retrieval:
  top_k: 5
  default_corpus: default
```

The embedding model is loaded lazily, on the first request that needs it (document ingestion or a
RAG-enabled chat) — not at gateway startup. That first request will be slow (model download + load);
everything after it reuses the already-loaded model.

> If `nvidia/Nemotron-3-Embed-8B-BF16` isn't recognized by `sentence-transformers` directly, check its
> model card for any `trust_remote_code` requirement — `SentenceTransformersEmbedder` passes the model
> name straight through to `sentence_transformers.SentenceTransformer(...)`
> (`app/services/rag/embeddings/sentence_transformers_embedder.py`), so anything that constructor accepts
> works here without code changes.

## 4. Start the gateway

The embedding model downloads in-process (not inside a Docker container), via the
`sentence-transformers`/`huggingface_hub` libraries, so make sure `HF_TOKEN` is set in **this** shell too
— it's picked up automatically, no code change needed:

```bash
export HF_TOKEN=hf_xxx  # if not already set in this shell
uv sync
uv run uvicorn app.main:app --port 9000
curl http://localhost:9000/health
```

## 5. Ingest a document

```bash
curl -X POST http://localhost:9000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The aie-local-llm gateway caps vLLM at 55% GPU memory utilization in this tutorial so an 8B embedding model can also run on GPU.",
    "corpus": "docs"
  }'
```

This triggers the embedding model's first load. Watch GPU memory climb during this call:

```bash
watch -n 1 nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

If this step OOMs, lower `VLLM_GPU_MEMORY_UTILIZATION` further and relaunch vLLM, or switch
`embeddings.device` back to `cpu` in `config/rag.yaml` and restart the gateway.

## 6. Ask a RAG-enabled question

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3.8-27B-NVFP4",
    "messages": [{"role": "user", "content": "What GPU memory utilization does this tutorial use for vLLM, and why?"}],
    "max_tokens": 100,
    "temperature": 0,
    "rag": {"enabled": true, "corpus": "docs"}
  }'
```

The response's `sources` field shows the retrieved chunk(s) that were injected as context — confirmation
that retrieval, not just the model's own knowledge, produced the answer.

## 7. (Optional) try the UI

```bash
uv sync --extra ui
uv run streamlit run ui/app.py --server.port 8502
```

Toggle "Use retrieval-augmented generation" on, set the corpus to `docs`, and ask the same question.

## 8. Clean up

```bash
bash docker/stop.sh
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| CUDA OOM when the embedding model loads | GPU headroom left by vLLM is smaller than the embedder needs | Lower `VLLM_GPU_MEMORY_UTILIZATION` and relaunch vLLM, or set `embeddings.device: cpu` |
| vLLM fails to start with an NVFP4-related error | `--quantization modelopt_fp4` missing or vLLM image predates NVFP4 support | Confirm the flag is set; check `docker logs aie-local-llm-vllm` |
| First ingest/chat call is very slow | Expected — it's downloading and loading the 8B embedding model | Subsequent calls reuse the loaded model and are fast |
| `/documents` or RAG-enabled chat returns empty `sources` | Query embedding didn't match any ingested chunk, or wrong `corpus` name | Confirm with `curl http://localhost:9000/documents?corpus=docs` that the chunk was actually ingested |
