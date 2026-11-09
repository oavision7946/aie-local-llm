#!/bin/bash
# Launch SGLang's OpenAI-compatible server on this DGX Spark.
# The server listens on 0.0.0.0 inside the container; the host side is published on 127.0.0.1 only.
# Runs as the calling user (not root), so files the engine writes to the shared Hugging Face
# cache stay owned by you. HOME=/tmp gives compile and JIT caches a writable place.
set -euo pipefail

# Pinned by digest so results are reproducible; override SGLANG_IMAGE to try another build.
SGLANG_IMAGE=${SGLANG_IMAGE:-"lmsysorg/sglang:v0.5.23@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"}
SGLANG_MODEL=${SGLANG_MODEL:-"Qwen/Qwen2.5-3B-Instruct"}
SGLANG_PORT=${SGLANG_PORT:-30000}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-4096}
# Fraction of the whole unified memory pool reserved for weights plus KV cache (see the vLLM script).
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.4}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-flashinfer}
# Extra flags appended as-is, e.g. SGLANG_EXTRA_ARGS="--tool-call-parser qwen3_coder".
SGLANG_EXTRA_ARGS=${SGLANG_EXTRA_ARGS:-}

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Warning: HF_TOKEN is not set — Hugging Face downloads will be unauthenticated" \
    "(lower rate limits, slower). Set it with: export HF_TOKEN=hf_xxx" \
    "(https://huggingface.co/docs/hub/en/security-tokens)" >&2
fi

docker run -d --rm --name aie-local-llm-sglang \
  --gpus all --ipc=host \
  -p "127.0.0.1:${SGLANG_PORT}:30000" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e USER="$(id -un)" \
  -e HF_HOME=/hf-cache \
  -v "${HOME}/.cache/huggingface:/hf-cache" \
  -e HF_TOKEN \
  "${SGLANG_IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${SGLANG_MODEL}" \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --context-length "${SGLANG_CONTEXT_LENGTH}" \
  --attention-backend "${SGLANG_ATTENTION_BACKEND}" \
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
  ${SGLANG_EXTRA_ARGS}

echo "SGLang starting on 127.0.0.1:${SGLANG_PORT} with model ${SGLANG_MODEL}."
echo "Check readiness with: curl http://127.0.0.1:${SGLANG_PORT}/health"
docker ps --filter name=^aie-local-llm-sglang$ --format 'Published ports: {{.Ports}}'
