from __future__ import annotations

import json
import logging

from llm_quant.cli.common import config_parser
from llm_quant.config.loader import load_config
from llm_quant.logging_utils import configure_logging
from llm_quant.orchestration.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = config_parser("Run full quantization + benchmark + quality pipeline.")
    parser.add_argument("--dry-run-quantize", action="store_true", help="Skip executing quantize binary.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg, verbose=args.verbose)
    logger.info("Pipeline CLI started")
    result = run_pipeline(args.config, dry_run_quantize=args.dry_run_quantize)
    logger.info("Pipeline CLI completed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
