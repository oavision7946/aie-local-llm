#!/bin/bash
# Launch vLLM's OpenAI-compatible server on this DGX Spark.
# The port is published on 127.0.0.1 only (see "Running engines safely" in the README).
# Runs as the calling user (not root), so files the engine writes to the shared Hugging Face
# cache stay owned by you. HOME=/tmp gives compile and JIT caches a writable place.
set -euo pipefail

# Pinned by digest so results are reproducible; override VLLM_IMAGE to try another build.
VLLM_IMAGE=${VLLM_IMAGE:-"vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34"}
VLLM_MODEL=${VLLM_MODEL:-"Qwen/Qwen2.5-3B-Instruct"}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
# Fraction of the whole unified memory pool (shared with the CPU and every other service on this
# machine) that vLLM reserves for weights plus KV cache. Size it from your memory budget.
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.4}
# Extra flags appended as-is, e.g. VLLM_EXTRA_ARGS="--quantization modelopt_fp4" for an NVFP4 checkpoint.
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Warning: HF_TOKEN is not set — Hugging Face downloads will be unauthenticated" \
    "(lower rate limits, slower). Set it with: export HF_TOKEN=hf_xxx" \
    "(https://huggingface.co/docs/hub/en/security-tokens)" >&2
fi

docker run -d --rm --name aie-local-llm-vllm \
  --gpus all --ipc=host \
  -p "127.0.0.1:${VLLM_PORT}:8000" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e USER="$(id -un)" \
  -e HF_HOME=/hf-cache \
  -v "${HOME}/.cache/huggingface:/hf-cache" \
  -e HF_TOKEN \
  "${VLLM_IMAGE}" \
  "${VLLM_MODEL}" \
  --dtype bfloat16 \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --served-model-name "${VLLM_MODEL}" \
  ${VLLM_EXTRA_ARGS}

echo "vLLM starting on 127.0.0.1:${VLLM_PORT} with model ${VLLM_MODEL}."
echo "Check readiness with: curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:${VLLM_PORT}/health"
docker ps --filter name=^aie-local-llm-vllm$ --format 'Published ports: {{.Ports}}'
