from __future__ import annotations

from llm_quant.cli.common import config_parser
from llm_quant.config.loader import load_config
from llm_quant.core.quantization import quantize_variants
from llm_quant.logging_utils import configure_logging


def main() -> None:
    parser = config_parser("Run llama.cpp quantization for configured variants.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running quantization binary.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs and stream quantize output.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg, verbose=args.verbose)
    results = quantize_variants(
        quantize_bin=cfg["paths"]["llama_quantize_bin"],
        source_model=cfg["paths"]["source_gguf"],
        output_dir=cfg["paths"]["quantized_models_dir"],
        variants=cfg["quantization"]["variants"],
        dry_run=args.dry_run,
        verbose=args.verbose,
        allow_requantize_fallback=bool(cfg["quantization"].get("allow_requantize_fallback", True)),
    )

    for row in results:
        print(f"{row['variant']}: {row['method']} -> {row['output_model']} [{row['status']}]")


if __name__ == "__main__":
    main()
