#!/usr/bin/env bash
set -euo pipefail

echo "[setup] action=prepare-model"
echo "[setup] config=config/config.yaml"
python3 -m llm_quant.cli.setup --verbose --config config/config.yaml prepare-model "$@"
echo "[setup] action complete: prepare-model"
