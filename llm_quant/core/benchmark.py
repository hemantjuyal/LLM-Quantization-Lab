from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Iterable, List

import psutil

from llm_quant.core.ollama_client import OllamaClient, OllamaError


logger = logging.getLogger(__name__)


def _extract_perf_fields(generate_response: dict) -> dict:
    eval_count = generate_response.get("eval_count") or 0
    eval_duration_ns = generate_response.get("eval_duration") or 0
    tokens_per_sec = 0.0
    if eval_duration_ns:
        tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
    return {
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "tokens_per_sec": tokens_per_sec,
    }


def _extract_model_memory_mb(ps_response: dict, model_name: str) -> float:
    models = ps_response.get("models", [])
    for model in models:
        if model.get("name") == model_name:
            size_bytes = model.get("size") or 0
            return size_bytes / (1024 * 1024)
    return 0.0


def run_benchmark(
    client: OllamaClient,
    models: Iterable[str],
    prompts: Iterable[str],
    repeats: int = 1,
) -> List[dict]:
    models = list(models)
    prompts = list(prompts)
    logger.info("Benchmark started: models=%s repeats=%d prompts=%d", models, repeats, len(prompts))
    process = psutil.Process()
    rows: List[dict] = []
    for model_name in models:
        for prompt in prompts:
            for run_idx in range(1, repeats + 1):
                logger.info("Benchmark run: model=%s run=%d", model_name, run_idx)
                mem_before_mb = process.memory_info().rss / (1024 * 1024)
                start = time.perf_counter()
                response = client.generate(model=model_name, prompt=prompt)
                latency_sec = time.perf_counter() - start
                mem_after_mb = process.memory_info().rss / (1024 * 1024)
                perf = _extract_perf_fields(response)

                model_mem_mb = 0.0
                try:
                    ps_data = client.ps()
                    model_mem_mb = _extract_model_memory_mb(ps_data, model_name)
                except OllamaError:
                    logger.warning("Could not collect Ollama memory for model=%s", model_name)
                    pass

                rows.append(
                    {
                        "model": model_name,
                        "run_id": run_idx,
                        "prompt": prompt,
                        "latency_sec": latency_sec,
                        "tokens_per_sec": perf["tokens_per_sec"],
                        "eval_count": perf["eval_count"],
                        "eval_duration_ns": perf["eval_duration_ns"],
                        "client_mem_delta_mb": mem_after_mb - mem_before_mb,
                        "ollama_model_mem_mb": model_mem_mb,
                        "response": response.get("response", ""),
                    }
                )
    logger.info("Benchmark completed: rows=%d", len(rows))
    return rows


def save_benchmark_results(rows: List[dict], output_csv: str | Path, output_json: str | Path) -> None:
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        if not rows:
            fh.write("")
        else:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    logger.info("Benchmark artifacts saved: csv=%s json=%s", csv_path, json_path)
