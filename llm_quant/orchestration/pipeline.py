from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Dict, List

from llm_quant.core.benchmark import run_benchmark, save_benchmark_results
from llm_quant.config.loader import load_config
from llm_quant.core.ollama_client import OllamaClient
from llm_quant.core.quality import evaluate_quality, save_quality_results, summarize_quality
from llm_quant.core.quantization import quantize_variants

logger = logging.getLogger(__name__)


def _avg(rows: List[dict], key: str, model_name: str) -> float:
    values = [float(r[key]) for r in rows if r.get("model") == model_name]
    return mean(values) if values else 0.0


def _evaluate_success(benchmark_rows: List[dict], quality_summary: Dict[str, float], cfg: Dict) -> Dict:
    success_cfg = cfg["success_criteria"]
    baseline = cfg["benchmark"]["baseline_model"]
    candidate = cfg["benchmark"]["candidate_model"]

    baseline_latency = _avg(benchmark_rows, "latency_sec", baseline)
    candidate_latency = _avg(benchmark_rows, "latency_sec", candidate)
    latency_reduction = 0.0
    if baseline_latency:
        latency_reduction = (baseline_latency - candidate_latency) / baseline_latency

    baseline_mem = _avg(benchmark_rows, "ollama_model_mem_mb", baseline)
    candidate_mem = _avg(benchmark_rows, "ollama_model_mem_mb", candidate)
    memory_reduction = 0.0
    if baseline_mem:
        memory_reduction = (baseline_mem - candidate_mem) / baseline_mem

    quality_floor = float(success_cfg["min_quality_score"])
    quality_value = float(quality_summary.get(candidate, 0.0))

    result = {
        "baseline_model": baseline,
        "candidate_model": candidate,
        "latency_reduction": latency_reduction,
        "memory_reduction": memory_reduction,
        "candidate_quality_score": quality_value,
        "targets": {
            "min_latency_reduction": float(success_cfg["min_latency_reduction"]),
            "min_memory_reduction": float(success_cfg["min_memory_reduction"]),
            "min_quality_score": quality_floor,
        },
    }
    result["is_success"] = (
        latency_reduction >= result["targets"]["min_latency_reduction"]
        and memory_reduction >= result["targets"]["min_memory_reduction"]
        and quality_value >= quality_floor
    )
    return result


def run_pipeline(config_path: str | Path, dry_run_quantize: bool = False) -> Dict:
    logger.info("Pipeline started: config=%s dry_run_quantize=%s", config_path, dry_run_quantize)
    cfg = load_config(config_path)

    paths = cfg["paths"]
    quant_cfg = cfg["quantization"]
    ollama_cfg = cfg["ollama"]
    bench_cfg = cfg["benchmark"]
    quality_cfg = cfg["quality"]

    quant_results = quantize_variants(
        quantize_bin=paths["llama_quantize_bin"],
        source_model=paths["source_gguf"],
        output_dir=paths["quantized_models_dir"],
        variants=quant_cfg["variants"],
        dry_run=dry_run_quantize,
        allow_requantize_fallback=bool(quant_cfg.get("allow_requantize_fallback", True)),
    )
    logger.info("Pipeline quantization stage complete: variants=%d", len(quant_results))

    client = OllamaClient(base_url=ollama_cfg["base_url"], timeout_sec=int(ollama_cfg["timeout_sec"]), default_stream=bool(ollama_cfg.get("stream", True)))

    benchmark_rows = run_benchmark(
        client=client,
        models=bench_cfg["models"],
        prompts=bench_cfg["prompts"],
        repeats=int(bench_cfg["repeats"]),
    )
    save_benchmark_results(
        benchmark_rows,
        output_csv=paths["benchmark_csv"],
        output_json=paths["benchmark_json"],
    )
    logger.info("Pipeline benchmark stage complete: rows=%d", len(benchmark_rows))

    quality_rows = evaluate_quality(
        client=client,
        reference_model=quality_cfg["reference_model"],
        candidate_models=quality_cfg["candidate_models"],
        prompts=quality_cfg["prompts"],
    )
    quality_summary = summarize_quality(quality_rows)
    save_quality_results(
        quality_rows,
        quality_summary,
        output_csv=paths["quality_csv"],
        output_json=paths["quality_json"],
    )
    logger.info("Pipeline quality stage complete: rows=%d", len(quality_rows))

    success = _evaluate_success(benchmark_rows, quality_summary, cfg)
    report_path = Path(paths["summary_json"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "quantization": quant_results,
                "quality_summary": quality_summary,
                "success": success,
            },
            fh,
            indent=2,
        )
    summary_csv_path = Path(paths["summary_csv"])
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "baseline_model",
                "candidate_model",
                "latency_reduction",
                "memory_reduction",
                "candidate_quality_score",
                "target_min_latency_reduction",
                "target_min_memory_reduction",
                "target_min_quality_score",
                "is_success",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "baseline_model": success["baseline_model"],
                "candidate_model": success["candidate_model"],
                "latency_reduction": success["latency_reduction"],
                "memory_reduction": success["memory_reduction"],
                "candidate_quality_score": success["candidate_quality_score"],
                "target_min_latency_reduction": success["targets"]["min_latency_reduction"],
                "target_min_memory_reduction": success["targets"]["min_memory_reduction"],
                "target_min_quality_score": success["targets"]["min_quality_score"],
                "is_success": success["is_success"],
            }
        )
    logger.info("Pipeline summary written: json=%s csv=%s", report_path, summary_csv_path)

    result = {
        "quantization": quant_results,
        "benchmark_rows": len(benchmark_rows),
        "quality_rows": len(quality_rows),
        "summary_path": str(report_path),
        "summary_csv_path": str(summary_csv_path),
        "success": success,
    }
    logger.info("Pipeline completed: success=%s", success.get("is_success"))
    return result
