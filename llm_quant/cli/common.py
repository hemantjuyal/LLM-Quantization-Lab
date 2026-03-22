from __future__ import annotations

import argparse


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration YAML file.",
    )
    return parser
