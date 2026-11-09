#!/bin/bash
# Launch vLLM with LMCache-accelerated KV cache offload on this DGX Spark.
# The pinned vLLM image already includes LMCache (lmcache 0.5.6 in vllm/vllm-openai:v0.32.0).
# The port is published on 127.0.0.1 only (see "Running engines safely" in the README).
set -euo pipefail

VLLM_LMCACHE_IMAGE=${VLLM_LMCACHE_IMAGE:-"vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34"}
VLLM_MODEL=${VLLM_MODEL:-"Qwen/Qwen2.5-3B-Instruct"}
VLLM_PORT=${VLLM_PORT:-8001}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
# Fraction of the whole unified memory pool reserved for weights plus KV cache (see docker/vllm/run.sh).
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.4}
# CPU-side KV cache size in GB. On unified memory this comes out of the same pool, so count it too.
LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_MAX_LOCAL_CPU_SIZE:-20}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Warning: HF_TOKEN is not set — Hugging Face downloads will be unauthenticated" \
    "(lower rate limits, slower). Set it with: export HF_TOKEN=hf_xxx" \
    "(https://huggingface.co/docs/hub/en/security-tokens)" >&2
fi

docker run -d --rm --name aie-local-llm-vllm-lmcache \
  --gpus all --ipc=host \
  -p "127.0.0.1:${VLLM_PORT}:8000" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN \
  -e LMCACHE_CHUNK_SIZE=256 \
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE}" \
  --entrypoint vllm \
  "${VLLM_LMCACHE_IMAGE}" \
  serve "${VLLM_MODEL}" \
  --dtype bfloat16 \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --served-model-name "${VLLM_MODEL}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  ${VLLM_EXTRA_ARGS}

echo "vLLM+LMCache starting on 127.0.0.1:${VLLM_PORT} with model ${VLLM_MODEL}."
echo "Check readiness with: curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:${VLLM_PORT}/health"
docker ps --filter name=^aie-local-llm-vllm-lmcache$ --format 'Published ports: {{.Ports}}'
