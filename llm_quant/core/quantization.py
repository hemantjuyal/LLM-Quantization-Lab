from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List


class QuantizationError(RuntimeError):
    """Raised when quantization command fails."""


logger = logging.getLogger(__name__)


def build_quantize_cmd(
    quantize_bin: str | Path,
    source_model: str | Path,
    output_model: str | Path,
    method: str,
    allow_requantize: bool = False,
) -> List[str]:
    cmd: List[str] = [str(quantize_bin)]
    if allow_requantize:
        cmd.extend(["--allow-requantize", "--leave-output-tensor"])
    cmd.extend([str(source_model), str(output_model), method])
    return cmd


def quantize_variants(
    quantize_bin: str | Path,
    source_model: str | Path,
    output_dir: str | Path,
    variants: Iterable[Dict[str, str]],
    dry_run: bool = False,
    verbose: bool = False,
    allow_requantize_fallback: bool = True,
) -> List[Dict[str, str]]:
    logger.info("Quantization started: source=%s output_dir=%s dry_run=%s", source_model, output_dir, dry_run)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, str]] = []
    for variant in variants:
        name = variant["name"]
        method = variant["method"]
        filename = variant["filename"]
        target = output_path / filename

        cmd = build_quantize_cmd(quantize_bin, source_model, target, method, allow_requantize=False)
        logger.info("Quantizing variant=%s method=%s output=%s", name, method, target)
        if not dry_run:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if verbose and proc.stdout:
                print(proc.stdout, end="")
            if verbose and proc.stderr:
                print(proc.stderr, end="")
            if proc.returncode != 0:
                if allow_requantize_fallback:
                    logger.warning(
                        "Initial quantization failed. Retrying with --allow-requantize --leave-output-tensor"
                    )
                    retry_cmd = build_quantize_cmd(
                        quantize_bin,
                        source_model,
                        target,
                        method,
                        allow_requantize=True,
                    )
                    proc = subprocess.run(retry_cmd, capture_output=True, text=True)
                    if verbose and proc.stdout:
                        print(proc.stdout, end="")
                    if verbose and proc.stderr:
                        print(proc.stderr, end="")

                if proc.returncode != 0:
                    error_text = proc.stderr.strip() or proc.stdout.strip()
                    if verbose and not error_text:
                        raise QuantizationError(f"Quantization failed for {name} ({method})")
                    raise QuantizationError(f"Quantization failed for {name} ({method}):\n{error_text}")

        results.append(
            {
                "variant": name,
                "method": method,
                "source_model": str(source_model),
                "output_model": str(target),
                "status": "dry_run" if dry_run else "ok",
            }
        )

    logger.info("Quantization completed: variants=%d", len(results))
    return results
