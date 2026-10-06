#!/bin/bash
# Launch SGLang's OpenAI-compatible server on this DGX Spark.
# Validated on this box against Qwen/Qwen2.5-3B-Instruct (see ~/workspace/inference/basic/).
set -euo pipefail

SGLANG_MODEL=${SGLANG_MODEL:-"Qwen/Qwen2.5-3B-Instruct"}
SGLANG_PORT=${SGLANG_PORT:-30000}
SGLANG_IMAGE=${SGLANG_IMAGE:-"lmsysorg/sglang:latest"}

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Warning: HF_TOKEN is not set — Hugging Face downloads will be unauthenticated" \
    "(lower rate limits, slower). Set it with: export HF_TOKEN=hf_xxx" \
    "(https://huggingface.co/docs/hub/en/security-tokens)" >&2
fi

docker run -d --rm --name aie-local-llm-sglang \
  --gpus all --ipc=host \
  -p "${SGLANG_PORT}:30000" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${SGLANG_IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${SGLANG_MODEL}" \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --context-length 4096 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.75

echo "SGLang starting on port ${SGLANG_PORT} with model ${SGLANG_MODEL}."
echo "Check readiness with: curl http://localhost:${SGLANG_PORT}/health"
