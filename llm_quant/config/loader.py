from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(RuntimeError):
    """Raised for invalid configuration."""


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ConfigError("Root config must be a mapping.")

    required = ["setup", "paths", "quantization", "ollama", "benchmark", "quality", "success_criteria"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigError(f"Missing required config sections: {missing}")

    for variant in data.get("quantization", {}).get("variants", []):
        for field in ("name", "method", "filename", "ollama_model", "modelfile"):
            if field not in variant:
                raise ConfigError(f"Missing quantization variant field: {field}")
        variant["modelfile"] = _resolve_to_project_root(cfg_path, variant["modelfile"])

    data["setup"]["llama_cpp"]["source_dir"] = _resolve_to_project_root(
        cfg_path, data["setup"]["llama_cpp"]["source_dir"]
    )
    data["setup"]["llama_cpp"]["build_dir"] = _resolve_to_project_root(
        cfg_path, data["setup"]["llama_cpp"]["build_dir"]
    )
    data["setup"]["model"]["exported_source_gguf"] = _resolve_to_project_root(
        cfg_path, data["setup"]["model"]["exported_source_gguf"]
    )

    for key, value in list(data["paths"].items()):
        data["paths"][key] = _resolve_to_project_root(cfg_path, value)
    if "logging" in data and isinstance(data["logging"], dict) and data["logging"].get("file"):
        data["logging"]["file"] = _resolve_to_project_root(cfg_path, data["logging"]["file"])

    return data


def _resolve_to_project_root(cfg_path: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    project_root = cfg_path.resolve().parent.parent
    return str((project_root / path).resolve())
