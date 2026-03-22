#!/usr/bin/env bash
set -euo pipefail

echo "[setup] action=bootstrap"
echo "[setup] config=config/config.yaml"
python3 -m llm_quant.cli.setup --verbose --config config/config.yaml bootstrap "$@"
echo "[setup] action complete: bootstrap"
