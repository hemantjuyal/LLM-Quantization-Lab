#!/usr/bin/env bash
set -euo pipefail

echo "[setup] action=setup-llama-cpp"
echo "[setup] config=config/config.yaml"
python3 -m llm_quant.cli.setup --verbose --config config/config.yaml setup-llama-cpp "$@"
echo "[setup] action complete: setup-llama-cpp"
