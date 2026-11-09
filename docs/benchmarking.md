# Benchmarking inference engines

`bench/serve_bench.py` measures an engine through its OpenAI-compatible API, so the same command works
for vLLM, SGLang, llama.cpp and anything else that speaks `/v1/chat/completions`. All prompts are
synthetic (generated log-like lines from a fixed seed); nothing real is sent or stored.

## What it measures

| Test | What it does | Reported as |
|:--|:--|:--|
| `decode` | 3 streaming requests of 256 tokens (`ignore_eos`), after a warm-up | median decode tokens/s, excluding the first token |
| `concurrency` | `--concurrency` (default 2) streams at once | mean per-stream tokens/s and aggregate tokens/s |
| `prefill` | one request per length in `--prefill-lengths`, each with a unique first line so a prefix cache cannot serve it | time to first token (TTFT) and prompt tokens/s |
| `tools` | one request with 3 tools that asks for all three to be called in parallel | pass if each tool is called exactly once with arguments that match its JSON schema |
| `json` | 3 requests with `response_format: {type: json_schema, strict: true}` | pass rate; output is validated against the schema |

Thinking-capable models are measured with thinking off by default (`chat_template_kwargs:
{"enable_thinking": false}`); add `--thinking` to run `tools` and `json` with thinking on.

## Workflow

```bash
# 1. Optional: drop the model's files from the page cache so the load is a true cold start.
uv run python bench/serve_bench.py evict ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8

# 2. Start an engine and time it until /v1/models answers.
t0=$(date +%s.%N)
VLLM_MODEL=Qwen/Qwen3.8-27B-FP8 VLLM_MAX_MODEL_LEN=262144 VLLM_GPU_MEMORY_UTILIZATION=0.35 \
VLLM_EXTRA_ARGS="--kv-cache-dtype fp8 --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --language-model-only" \
  bash docker/vllm/run.sh
uv run python bench/serve_bench.py wait --base-url http://127.0.0.1:8000/v1 --started-at "$t0"

# 3. Run the suite. Results go to bench/results/<label>.json and a row in bench/results/results.csv.
uv run python bench/serve_bench.py run --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.8-27B-FP8 --label vllm-fp8 --container aie-local-llm-vllm \
  --prefill-lengths 8192,32768,131072,250000

bash docker/stop.sh
```

Record the memory setting (`--gpu-memory-utilization`, `--mem-fraction-static` or llama.cpp's
`--ctx-size`) with every result: it decides how much context fits. On this machine, `docker stats`
does not include memory allocated on the GPU, so the script also records `free -m`.

## Results: Qwen3.8-27B on DGX Spark (November 2026)

Setup: one DGX Spark (GB10, 128 GB unified memory, 273 GB/s), images pinned as in `docker/`, model
`Qwen/Qwen3.8-27B` in the variants below, an 8-bit KV cache everywhere (`fp8` / `fp8_e4m3` / `q8_0`),
memory fraction 0.35 for vLLM and SGLang (about 42 GB of the pool), thinking off unless noted, seed 1234.
Cold load means the model files were evicted from the page cache first.

Single-stream decode is bounded by memory bandwidth divided by the weight size: about 9-10 tokens/s
for the 8-bit variants and 12-14 tokens/s for the 4-bit ones.

| Engine | Weights | Cold load | Max context at 0.35 | Decode tok/s | 2 streams: per stream / total | Prefill tok/s at 8K / 32K | TTFT at ~128K | Tools | JSON |
|:--|:--|--:|:--|--:|:--|:--|--:|:-:|:-:|
| vLLM 0.32.0 | FP8 (Qwen) | 403 s | 262,144 (405K-token pool) | 8.15 | 8.22 / 16.4 | 1,994 / 1,725 | 124 s | pass | 3/3 |
| vLLM 0.32.0 | NVFP4 (NVIDIA) | 371 s | 262,144 (574K-token pool) | **12.32** | 11.89 / 23.7 | **2,818 / 2,279** | 106 s | pass | 3/3 |
| vLLM 0.31 (NVIDIA NGC 26.10) | NVFP4 (NVIDIA) | 281 s | 262,144 (434K-token pool) | 12.08 | 11.66 / 23.2 | 2,814 / 2,285 | 104 s | pass | 3/3 |
| SGLang 0.5.23 | FP8 (Qwen) | 509 s | 134,318-token pool | 7.95 | 8.07 / 16.1 | 860 / 797 | 189 s (120K) | pass | 3/3 |
| llama.cpp b12199 | GGUF Q8_0 | **31 s** | 131,072 per slot, 2 slots | 8.07 | 7.81 / 15.5 | 858 / 805 | 192 s (120K) | pass | 3/3 |
| llama.cpp b12199 | GGUF Q4_K_M | **21 s** | 131,072 per slot, 2 slots | 11.02 | 10.37 / 20.4 | 937 / 873 | not run | pass | 3/3 |
| vLLM 0.32.0 + LMCache | NVFP4 (NVIDIA) | fails to start (see below) | | | | | | | |

Observations:

- All engines reach 80-90% of the bandwidth ceiling for decode; the engine matters much more for
  **prefill**, where vLLM is 2-3x faster than SGLang and llama.cpp at the settings above.
- Qwen3.8's hybrid attention keeps a KV cache in only 16 of its 64 layers (64 KiB per token in 16-bit),
  so the full 262K context fits in a single 42 GB budget with vLLM. SGLang reserved more of the same
  budget for the linear-attention state and could offer only a 134K-token pool.
- llama.cpp loads an order of magnitude faster (no compilation or CUDA graph capture) and is a good fit
  for quick model swaps; vLLM and SGLang take 6-8 minutes to become ready from a cold cache.
- With thinking on, a JSON request can spend its whole `max_tokens` budget on reasoning and return
  empty content; give thinking requests a larger budget or lower `reasoning_effort`.
- vLLM's multi-token prediction (`--speculative-config '{"method":"mtp",...}'`) needs extra memory: at
  0.35 the maximum context drops to about 204K, and `--max-num-seqs` must be lowered (for example to 8)
  for the linear-attention state cache. Not benchmarked here.
- The upstream `vllm/vllm-openai:v0.32.0` image already includes LMCache 0.5.6, so
  `docker/vllm-lmcache/run.sh` uses it directly. With Qwen3.8 (hybrid attention), the LMCache connector
  fails at startup with `Failed to promote local KV cache specs to one unified type`, also with
  `--disable-hybrid-kv-cache-manager`. vLLM's built-in prefix caching still works for this model.
- NVIDIA's NGC vLLM container loads the same weights about 7x faster (20 s vs 143 s) and is ready 90 s
  sooner; at runtime it matches upstream vLLM.
- With 250K-token prompts, vLLM's TTFT is 350 s (FP8) and 312 s (NVFP4).

Raw results: [`bench/results/`](../bench/results/) (one JSON file per run plus `results.csv`).
