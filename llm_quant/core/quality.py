from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Iterable, List

from rapidfuzz.fuzz import ratio

from llm_quant.core.ollama_client import OllamaClient


logger = logging.getLogger(__name__)


def evaluate_quality(
    client: OllamaClient,
    reference_model: str,
    candidate_models: Iterable[str],
    prompts: Iterable[str],
) -> List[dict]:
    candidate_models = list(candidate_models)
    prompts = list(prompts)
    logger.info(
        "Quality evaluation started: reference=%s candidates=%s prompts=%d",
        reference_model,
        candidate_models,
        len(prompts),
    )
    rows: List[dict] = []
    for prompt in prompts:
        logger.debug("Quality prompt evaluation: prompt_chars=%d", len(prompt))
        reference = client.generate(model=reference_model, prompt=prompt).get("response", "")
        for candidate in candidate_models:
            candidate_resp = client.generate(model=candidate, prompt=prompt).get("response", "")
            score = ratio(reference, candidate_resp) / 100.0
            rows.append(
                {
                    "prompt": prompt,
                    "reference_model": reference_model,
                    "candidate_model": candidate,
                    "quality_score": score,
                    "reference_response": reference,
                    "candidate_response": candidate_resp,
                }
            )
    logger.info("Quality evaluation completed: rows=%d", len(rows))
    return rows


def summarize_quality(rows: List[dict]) -> dict:
    by_model: dict[str, List[float]] = {}
    for row in rows:
        by_model.setdefault(row["candidate_model"], []).append(row["quality_score"])

    summary = {model: mean(scores) if scores else 0.0 for model, scores in by_model.items()}
    logger.info("Quality summary computed for models=%s", list(summary.keys()))
    return summary


def save_quality_results(rows: List[dict], summary: dict, output_csv: str | Path, output_json: str | Path) -> None:
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            fh.write("")

    payload = {"rows": rows, "summary": summary}
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Quality artifacts saved: csv=%s json=%s", csv_path, json_path)
