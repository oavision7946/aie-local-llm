# Tutorial 1: Run a local LLM (Qwen3.5-9B)

This walks through serving a single local model — [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B)
— through vLLM, and talking to it via this project's OpenAI-compatible gateway. No RAG, no embedding
model — just the shortest path from zero to a working local LLM endpoint.

Qwen3.5-9B is a good first model for this box: 9B parameters, comfortably inside this DGX Spark's 128GB
of unified memory, and small enough to load in a couple of minutes.

## 0. Prerequisites

```bash
nvidia-smi          # confirm the GPU is visible
docker --version    # confirm Docker + the NVIDIA Container Toolkit are set up
```

Set an HF token so the download is authenticated — without one you'll see
`Warning: You are sending unauthenticated requests to the HF Hub` in the container logs, and get lower
rate limits and slower downloads (a token is also required if `Qwen/Qwen3.5-9B` is ever gated):

```bash
export HF_TOKEN=hf_xxx  # https://huggingface.co/docs/hub/en/security-tokens
```

`docker/vllm/run.sh` passes `HF_TOKEN` through to the container automatically if it's set in your shell.

## 1. Launch vLLM with this model

`Qwen/Qwen3.5-9B` is a *reasoning* model — by default it generates a full chain-of-thought
(`<think>...</think>`) before every answer, even for something like "12\*17". That's slower and, without
telling vLLM how to parse it, the thinking/answer boundary can come out garbled in the response. Launch
with `--reasoning-parser qwen3` so vLLM parses it correctly, and
`--default-chat-template-kwargs '{"enable_thinking": false}'` to skip the chain-of-thought entirely for
quick, direct answers (this is the model's own documented switch — see
[Qwen's vLLM deployment guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)):

```bash
cd aie-local-llm
VLLM_MODEL=Qwen/Qwen3.5-9B \
VLLM_EXTRA_ARGS='--reasoning-parser qwen3 --default-chat-template-kwargs {"enable_thinking":false}' \
  bash docker/vllm/run.sh
```

For harder questions where you want the model to actually reason, keep `--reasoning-parser qwen3` but
drop `--default-chat-template-kwargs ...` — the model will think by default, taking longer per request,
but `--reasoning-parser qwen3` still keeps the response clean: vLLM strips the `<think>...</think>` block
into its own `reasoning_content` field, so this gateway's `content` only ever gets the final answer
either way. (This gateway doesn't currently surface `reasoning_content` itself — just the clean final
answer in `content`.)

The first run downloads the model to `~/.cache/huggingface` (several GB) — expect this step to take a
few minutes depending on your network. Wait for it to come up:

```bash
until curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health | grep -q 200; do
  sleep 3
done
echo "vLLM is ready"
```

If it doesn't come up, check the logs:

```bash
docker logs aie-local-llm-vllm
```

## 2. Point the gateway at it

Edit `config/engines.yaml` so the `vllm` entry's `model` matches what you launched:

```yaml
active: vllm

engines:
  - name: vllm
    base_url: http://127.0.0.1:8000/v1
    model: Qwen/Qwen3.5-9B
    enabled: true
```

## 3. Start the gateway

```bash
uv sync
uv run uvicorn app.main:app --port 9000
```

```bash
curl http://localhost:9000/health
# {"status":"ok","active_engine":"vllm","model":"Qwen/Qwen3.5-9B"}

curl http://localhost:9000/v1/models
# {"object":"list","data":[{"id":"Qwen/Qwen3.5-9B", ...}]}
```

## 4. Ask it something

With thinking disabled (step 1), this should answer directly and quickly:

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-9B",
    "messages": [{"role": "user", "content": "What is 12*17? Answer with just the number, no explanation."}],
    "max_tokens": 50,
    "temperature": 0
  }'
```

This should return `"204"` in well under a second, with `"finish_reason": "stop"`.

> **Note:** `enable_thinking: false` only suppresses the `<think>...</think>` chain-of-thought block — it
> doesn't stop the model from being conversationally verbose. Drop the "answer with just the number"
> instruction and ask plainly `"What is 12*17?"`, and Qwen3.5-9B will still walk through the arithmetic
> step by step in its final answer (that's normal model behavior, not a thinking leak — there's still no
> `<think>` tag in it). With only `max_tokens: 50` that walkthrough gets cut off mid-explanation
> (`"finish_reason": "length"`); either raise `max_tokens` to ~200 to let it finish, or do what we did
> above and ask for a direct answer in the prompt.

Or point any OpenAI-client-compatible application at `http://<this-host>:9000/v1` — this is the whole
point of the gateway being OpenAI-compatible.

## 5. (Optional) try the UI

```bash
uv sync --extra ui
uv run streamlit run ui/app.py --server.port 8502
```

Open `http://localhost:8502`, pick the model, and chat (leave "Use retrieval-augmented generation" off —
there's no RAG in this tutorial).

## 6. Clean up

```bash
bash docker/stop.sh
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker logs` shows a CUDA OOM during model load | Another engine container is still running and holding GPU memory | `docker ps` / `bash docker/stop.sh`, then retry |
| vLLM fails to recognize the model's architecture | Pinned vLLM image predates this model | Pull a newer `vllm/vllm-openai` tag and update `docker/vllm/run.sh` |
| `/health` never returns 200 | Model still downloading/loading | `docker logs -f aie-local-llm-vllm` and wait; a 9B model typically takes 1-3 minutes to load once downloaded |
| Gateway's `/v1/chat/completions` returns 502 | `config/engines.yaml`'s `model` doesn't match what vLLM was launched with, or the container isn't up | Confirm with `curl http://localhost:8000/v1/models` directly against vLLM |
| `docker logs` shows "sending unauthenticated requests to the HF Hub" | `HF_TOKEN` wasn't set before launching | `export HF_TOKEN=hf_xxx`, then `bash docker/stop.sh` and relaunch |
| Answer is slow and full of `<think>...</think>` reasoning text | Qwen3.5-9B reasons by default | Relaunch with `--default-chat-template-kwargs '{"enable_thinking":false}'` (step 1) |
| Answer is truncated or ends with garbled text (e.g. a stray `cw` near `</think>`) | vLLM was launched without `--reasoning-parser qwen3`, so it can't correctly split thinking from the final answer | Relaunch with `VLLM_EXTRA_ARGS` including `--reasoning-parser qwen3` (step 1) |
| `"finish_reason": "length"` and the answer stops mid-sentence, but there's no `<think>` tag anywhere | Thinking is correctly disabled, but Qwen3.5-9B still explains its reasoning in the final answer and ran out of `max_tokens` | Not a thinking leak — either raise `max_tokens` (~200 for simple arithmetic) or ask for a direct answer in the prompt, as in step 4 |

## Next step

[Tutorial 2](tutorial-2-llm-and-embeddings.md) adds an embedding model alongside the LLM so you can use
RAG, not just plain chat.
