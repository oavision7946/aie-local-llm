#!/bin/bash
# Launch vLLM's OpenAI-compatible server on this DGX Spark.
# Validated on this box against Qwen/Qwen2.5-3B-Instruct (see ~/workspace/inference/basic/).
set -euo pipefail

VLLM_MODEL=${VLLM_MODEL:-"Qwen/Qwen2.5-3B-Instruct"}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
# Lower this (e.g. 0.5) to leave GPU headroom for an embedding model running alongside vLLM.
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
# Extra flags appended as-is, e.g. VLLM_EXTRA_ARGS="--quantization modelopt_fp4" for an NVFP4 checkpoint.
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Warning: HF_TOKEN is not set — Hugging Face downloads will be unauthenticated" \
    "(lower rate limits, slower). Set it with: export HF_TOKEN=hf_xxx" \
    "(https://huggingface.co/docs/hub/en/security-tokens)" >&2
fi

docker run -d --rm --name aie-local-llm-vllm \
  --gpus all --ipc=host \
  -p "${VLLM_PORT}:8000" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  vllm/vllm-openai:v0.25.1 \
  "${VLLM_MODEL}" \
  --dtype bfloat16 \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --served-model-name "${VLLM_MODEL}" \
  ${VLLM_EXTRA_ARGS}

echo "vLLM starting on port ${VLLM_PORT} with model ${VLLM_MODEL}."
echo "Check readiness with: curl -s -o /dev/null -w '%{http_code}\n' http://localhost:${VLLM_PORT}/health"
