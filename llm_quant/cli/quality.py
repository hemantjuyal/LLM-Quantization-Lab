from __future__ import annotations

import logging

from llm_quant.cli.common import config_parser
from llm_quant.config.loader import load_config
from llm_quant.core.ollama_client import OllamaClient
from llm_quant.core.quality import evaluate_quality, save_quality_results, summarize_quality
from llm_quant.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = config_parser("Run quality evaluation against reference model.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg, verbose=args.verbose)
    logger.info("Quality CLI started")
    client = OllamaClient(
        cfg["ollama"]["base_url"],
        timeout_sec=int(cfg["ollama"]["timeout_sec"]),
        default_stream=bool(cfg["ollama"].get("stream", True)),
    )
    logger.info("Evaluating quality reference=%s candidates=%s", cfg["quality"]["reference_model"], cfg["quality"]["candidate_models"])
    rows = evaluate_quality(
        client=client,
        reference_model=cfg["quality"]["reference_model"],
        candidate_models=cfg["quality"]["candidate_models"],
        prompts=cfg["quality"]["prompts"],
    )
    summary = summarize_quality(rows)
    save_quality_results(
        rows=rows,
        summary=summary,
        output_csv=cfg["paths"]["quality_csv"],
        output_json=cfg["paths"]["quality_json"],
    )
    logger.info("Quality CLI completed")
    print("Quality summary:")
    for model, score in summary.items():
        print(f"- {model}: {score:.4f}")
    print(f"CSV: {cfg['paths']['quality_csv']}")
    print(f"JSON: {cfg['paths']['quality_json']}")


if __name__ == "__main__":
    main()
