#!/usr/bin/env bash
set -euo pipefail

echo "[quantize] action=quantize"
echo "[quantize] config=config/config.yaml"
python3 -m llm_quant.cli.quantize --verbose --config config/config.yaml "$@"
echo "[quantize] action complete: quantize"
