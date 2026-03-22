from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


class SetupError(RuntimeError):
    """Raised when project setup fails."""

logger = logging.getLogger(__name__)


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SetupError(f"Required binary not found in PATH: {name}")


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        logger.info(msg)
    else:
        logger.debug(msg)


def _run(cmd: List[str], cwd: str | Path | None = None, verbose: bool = False) -> None:
    location = f" (cwd={cwd})" if cwd else ""
    _log(f"$ {' '.join(cmd)}{location}", verbose)
    if verbose:
        proc = subprocess.run(cmd, cwd=cwd, text=True)
    else:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        if verbose:
            raise SetupError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
        stderr = proc.stderr.strip() or proc.stdout.strip()
        raise SetupError(f"Command failed: {' '.join(cmd)}\n{stderr}")


def setup_llama_cpp(cfg: Dict, rebuild: bool = False, verbose: bool = False) -> Dict:
    _log("Starting llama.cpp setup", verbose)
    _require_binary("git")
    _require_binary("cmake")

    setup_cfg = cfg["setup"]["llama_cpp"]
    source_dir = Path(setup_cfg["source_dir"])
    build_dir = Path(setup_cfg["build_dir"])
    repo_url = setup_cfg["repo_url"]
    ref = setup_cfg.get("ref", "master")

    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repo_url, str(source_dir)], verbose=verbose)

    _run(["git", "fetch", "--all", "--tags"], cwd=source_dir, verbose=verbose)
    _run(["git", "checkout", ref], cwd=source_dir, verbose=verbose)

    if rebuild and build_dir.exists():
        _log(f"Removing build dir: {build_dir}", verbose)
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)
    _run(["cmake", "-S", str(source_dir), "-B", str(build_dir)], verbose=verbose)
    _run(["cmake", "--build", str(build_dir), "--config", "Release", "-j"], verbose=verbose)

    quantize_bin = build_dir / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        raise SetupError(f"llama-quantize binary not found at {quantize_bin}")
    _log(f"llama.cpp setup complete. quantize_bin={quantize_bin}", verbose)

    return {
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "quantize_bin": str(quantize_bin),
    }


def _extract_blob_path_from_modelfile_text(modelfile_text: str) -> Path:
    for raw in modelfile_text.splitlines():
        line = raw.strip()
        if line.startswith("FROM "):
            return Path(line.replace("FROM ", "", 1).strip())
    raise SetupError("Could not locate FROM path in `ollama show --modelfile` output.")


def prepare_source_gguf(cfg: Dict, force: bool = False, verbose: bool = False) -> Dict:
    _log("Starting source GGUF preparation from Ollama", verbose)
    _require_binary("ollama")

    model_cfg = cfg["setup"]["model"]
    source_model = model_cfg["ollama_model"]
    output_path = Path(model_cfg["exported_source_gguf"])

    if output_path.exists() and not force:
        _log(f"GGUF already exists, skipping export: {output_path}", verbose)
        return {
            "source_model": source_model,
            "exported_source_gguf": str(output_path),
            "status": "exists",
        }

    if model_cfg.get("pull_first", True):
        _run(["ollama", "pull", source_model], verbose=verbose)

    _log(f"Inspecting model modelfile: {source_model}", verbose)
    show_proc = subprocess.run(
        ["ollama", "show", "--modelfile", source_model],
        text=True,
        capture_output=True,
    )
    if show_proc.returncode != 0:
        msg = show_proc.stderr.strip() or show_proc.stdout.strip()
        raise SetupError(f"Failed to inspect Ollama model {source_model}: {msg}")
    if verbose and show_proc.stdout:
        print(show_proc.stdout)

    blob_path = _extract_blob_path_from_modelfile_text(show_proc.stdout)
    if not blob_path.exists():
        raise SetupError(f"Ollama blob path does not exist: {blob_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Copying model blob to GGUF path: {output_path}", verbose)
    shutil.copy2(blob_path, output_path)

    return {
        "source_model": source_model,
        "blob_path": str(blob_path),
        "exported_source_gguf": str(output_path),
        "status": "exported",
    }


def generate_modelfiles(cfg: Dict, verbose: bool = False) -> List[Dict]:
    _log("Generating Modelfiles for quantized variants", verbose)
    quantized_dir = Path(cfg["paths"]["quantized_models_dir"])
    results: List[Dict] = []
    for variant in cfg["quantization"]["variants"]:
        model_file = quantized_dir / variant["filename"]
        modelfile_path = Path(variant["modelfile"])
        modelfile_path.parent.mkdir(parents=True, exist_ok=True)
        modelfile_path.write_text(f"FROM {model_file}\n", encoding="utf-8")
        _log(f"Wrote {modelfile_path} -> FROM {model_file}", verbose)
        results.append(
            {
                "variant": variant["name"],
                "modelfile": str(modelfile_path),
                "ollama_model": variant["ollama_model"],
            }
        )
    return results


def register_ollama_models(cfg: Dict, verbose: bool = False) -> List[Dict]:
    _log("Registering quantized models in Ollama", verbose)
    rows: List[Dict] = []
    for variant in cfg["quantization"]["variants"]:
        target_model = variant["ollama_model"]
        modelfile_path = variant["modelfile"]
        _run(["ollama", "create", target_model, "-f", modelfile_path], verbose=verbose)
        rows.append({"variant": variant["name"], "ollama_model": target_model, "status": "created"})
    return rows


def bootstrap_local_setup(
    cfg: Dict, rebuild_llama_cpp: bool = False, force_model_export: bool = False, verbose: bool = False
) -> Dict:
    _log("Running full bootstrap flow", verbose)
    llama_cpp_result = setup_llama_cpp(cfg, rebuild=rebuild_llama_cpp, verbose=verbose)
    gguf_result = prepare_source_gguf(cfg, force=force_model_export, verbose=verbose)
    modelfiles = generate_modelfiles(cfg, verbose=verbose)
    return {
        "llama_cpp": llama_cpp_result,
        "source_model": gguf_result,
        "modelfiles": modelfiles,
    }
