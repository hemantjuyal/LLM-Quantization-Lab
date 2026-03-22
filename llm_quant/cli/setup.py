from __future__ import annotations

import argparse
import json
import logging

from llm_quant.config.loader import load_config
from llm_quant.logging_utils import configure_logging
from llm_quant.setup import (
    bootstrap_local_setup,
    generate_modelfiles,
    prepare_source_gguf,
    register_ollama_models,
    setup_llama_cpp,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup utilities for local Mac LLM quantization workflow.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose setup logs.")
    parser.add_argument(
        "action",
        choices=[
            "setup-llama-cpp",
            "prepare-model",
            "generate-modelfiles",
            "register-ollama-models",
            "bootstrap",
        ],
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild llama.cpp from clean build dir.")
    parser.add_argument("--force", action="store_true", help="Force model export even if GGUF exists.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg, verbose=args.verbose)
    logging.getLogger(__name__).info("Starting setup action: %s", args.action)
    if args.action == "setup-llama-cpp":
        result = setup_llama_cpp(cfg, rebuild=args.rebuild, verbose=args.verbose)
    elif args.action == "prepare-model":
        result = prepare_source_gguf(cfg, force=args.force, verbose=args.verbose)
    elif args.action == "generate-modelfiles":
        result = generate_modelfiles(cfg, verbose=args.verbose)
    elif args.action == "register-ollama-models":
        result = register_ollama_models(cfg, verbose=args.verbose)
    else:
        result = bootstrap_local_setup(
            cfg,
            rebuild_llama_cpp=args.rebuild,
            force_model_export=args.force,
            verbose=args.verbose,
        )

    logging.getLogger(__name__).info("Completed setup action: %s", args.action)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
