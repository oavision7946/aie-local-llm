#!/bin/bash
# Stop any running aie-local-llm engine containers.
set -euo pipefail

for name in aie-local-llm-vllm aie-local-llm-vllm-lmcache aie-local-llm-sglang; do
  if docker ps -q --filter "name=^${name}$" | grep -q .; then
    echo "Stopping ${name}..."
    docker stop "${name}" >/dev/null
  fi
done
