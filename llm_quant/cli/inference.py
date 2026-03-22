from __future__ import annotations

import logging

from llm_quant.cli.common import config_parser
from llm_quant.config.loader import load_config
from llm_quant.core.ollama_client import OllamaClient
from llm_quant.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = config_parser("Run single prompt inference against Ollama.")
    parser.add_argument("--model", required=False, help="Ollama model name override.")
    parser.add_argument("--prompt", required=True, help="Prompt to run.")
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming in console.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg, verbose=args.verbose)
    model = args.model or cfg["benchmark"]["candidate_model"]
    stream_enabled = bool(cfg["ollama"].get("stream", True)) and not args.no_stream
    logger.info("Inference CLI started model=%s stream=%s", model, stream_enabled)
    client = OllamaClient(
        cfg["ollama"]["base_url"],
        timeout_sec=int(cfg["ollama"]["timeout_sec"]),
        default_stream=stream_enabled,
    )

    if stream_enabled:
        response = client.generate(model=model, prompt=args.prompt, on_token=lambda t: print(t, end="", flush=True))
        print()
    else:
        response = client.generate(model=model, prompt=args.prompt, stream=False)
        print(response.get("response", ""))
    logger.info("Inference CLI completed model=%s", model)


if __name__ == "__main__":
    main()
