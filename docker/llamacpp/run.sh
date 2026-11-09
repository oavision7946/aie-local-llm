#!/bin/bash
# Launch llama.cpp's OpenAI-compatible server (llama-server) for a GGUF model on this DGX Spark.
# The GGUF file is read from the shared Hugging Face cache; download it first, e.g.:
#   uvx --from 'huggingface_hub[cli]' hf download ggml-org/Qwen3.8-27B-GGUF Qwen3.8-27B-Q8_0.gguf
# The server listens on 0.0.0.0 inside the container; the host side is published on 127.0.0.1 only.
set -euo pipefail

# Pinned by digest so results are reproducible; override LLAMACPP_IMAGE to try another build.
LLAMACPP_IMAGE=${LLAMACPP_IMAGE:-"ghcr.io/ggml-org/llama.cpp:server-cuda13-b12199@sha256:19c9cdce4b2c28945e8eb5e10643a4d90876e918140210e18aebfc4ea7909536"}
LLAMACPP_HF_REPO=${LLAMACPP_HF_REPO:-"ggml-org/Qwen3.8-27B-GGUF"}
LLAMACPP_HF_FILE=${LLAMACPP_HF_FILE:-"Qwen3.8-27B-Q8_0.gguf"}
# Name clients use in the "model" field.
LLAMACPP_MODEL_ALIAS=${LLAMACPP_MODEL_ALIAS:-"${LLAMACPP_HF_REPO}:${LLAMACPP_HF_FILE%.gguf}"}
LLAMACPP_PORT=${LLAMACPP_PORT:-8002}
# Total context across all slots: each of the LLAMACPP_PARALLEL slots gets CTX_SIZE / PARALLEL tokens.
# llama.cpp allocates the whole KV cache up front, so this sets its memory use directly.
LLAMACPP_CTX_SIZE=${LLAMACPP_CTX_SIZE:-8192}
LLAMACPP_PARALLEL=${LLAMACPP_PARALLEL:-1}
LLAMACPP_EXTRA_ARGS=${LLAMACPP_EXTRA_ARGS:-}

HF_HOME_DIR="${HOME}/.cache/huggingface"
repo_dir="${HF_HOME_DIR}/hub/models--${LLAMACPP_HF_REPO//\//--}"
model_file=$(ls "${repo_dir}"/snapshots/*/"${LLAMACPP_HF_FILE}" 2>/dev/null | head -n 1 || true)
if [ -z "${model_file}" ]; then
  echo "GGUF file ${LLAMACPP_HF_FILE} from ${LLAMACPP_HF_REPO} not found under ${repo_dir}." >&2
  echo "Download it first (see the comment at the top of this script)." >&2
  exit 1
fi

docker run -d --rm --name aie-local-llm-llamacpp \
  --gpus all \
  -p "127.0.0.1:${LLAMACPP_PORT}:8080" \
  -v "${HF_HOME_DIR}:/hf:ro" \
  "${LLAMACPP_IMAGE}" \
  --host 0.0.0.0 \
  --port 8080 \
  --model "/hf/${model_file#"${HF_HOME_DIR}/"}" \
  --alias "${LLAMACPP_MODEL_ALIAS}" \
  --ctx-size "${LLAMACPP_CTX_SIZE}" \
  --parallel "${LLAMACPP_PARALLEL}" \
  --n-gpu-layers 999 \
  --flash-attn on \
  --jinja \
  ${LLAMACPP_EXTRA_ARGS}

echo "llama.cpp server starting on 127.0.0.1:${LLAMACPP_PORT} with ${LLAMACPP_HF_FILE}."
echo "Check readiness with: curl http://127.0.0.1:${LLAMACPP_PORT}/health"
docker ps --filter name=^aie-local-llm-llamacpp$ --format 'Published ports: {{.Ports}}'
