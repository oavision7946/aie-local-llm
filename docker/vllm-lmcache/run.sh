#!/bin/bash
# Launch vLLM with LMCache-accelerated KV cache offload on this DGX Spark.
# Uses the locally-built `lmcache-vllm:spark` image (vllm/vllm-openai:v0.25.1 + `pip install lmcache`).
set -euo pipefail

VLLM_MODEL=${VLLM_MODEL:-"Qwen/Qwen2.5-3B-Instruct"}
VLLM_PORT=${VLLM_PORT:-8001}

KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Warning: HF_TOKEN is not set — Hugging Face downloads will be unauthenticated" \
    "(lower rate limits, slower). Set it with: export HF_TOKEN=hf_xxx" \
    "(https://huggingface.co/docs/hub/en/security-tokens)" >&2
fi

docker run -d --rm --name aie-local-llm-vllm-lmcache \
  --gpus all --ipc=host \
  -p "${VLLM_PORT}:8000" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e LMCACHE_CHUNK_SIZE=256 \
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_MAX_LOCAL_CPU_SIZE=20 \
  --entrypoint vllm \
  lmcache-vllm:spark \
  serve "${VLLM_MODEL}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --served-model-name "${VLLM_MODEL}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}"

echo "vLLM+LMCache starting on port ${VLLM_PORT} with model ${VLLM_MODEL}."
echo "Check readiness with: curl -s -o /dev/null -w '%{http_code}\n' http://localhost:${VLLM_PORT}/health"
