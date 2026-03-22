from __future__ import annotations

import logging

from llm_quant.core.benchmark import run_benchmark, save_benchmark_results
from llm_quant.cli.common import config_parser
from llm_quant.config.loader import load_config
from llm_quant.core.ollama_client import OllamaClient
from llm_quant.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = config_parser("Run benchmark for configured Ollama models and prompts.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg, verbose=args.verbose)
    logger.info("Benchmark CLI started")
    client = OllamaClient(
        cfg["ollama"]["base_url"],
        timeout_sec=int(cfg["ollama"]["timeout_sec"]),
        default_stream=bool(cfg["ollama"].get("stream", True)),
    )
    logger.info("Running benchmark for models=%s", cfg["benchmark"]["models"])
    rows = run_benchmark(
        client=client,
        models=cfg["benchmark"]["models"],
        prompts=cfg["benchmark"]["prompts"],
        repeats=int(cfg["benchmark"]["repeats"]),
    )
    save_benchmark_results(
        rows,
        output_csv=cfg["paths"]["benchmark_csv"],
        output_json=cfg["paths"]["benchmark_json"],
    )
    logger.info("Benchmark CLI completed")
    print(f"Benchmark rows: {len(rows)}")
    print(f"CSV: {cfg['paths']['benchmark_csv']}")
    print(f"JSON: {cfg['paths']['benchmark_json']}")


if __name__ == "__main__":
    main()
