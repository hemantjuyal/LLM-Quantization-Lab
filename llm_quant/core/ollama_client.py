from __future__ import annotations

import json
import logging
from typing import Callable

import requests


class OllamaError(RuntimeError):
    """Raised when Ollama API request fails."""


class OllamaClient:
    def __init__(self, base_url: str, timeout_sec: int = 120, default_stream: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.default_stream = default_stream
        self.logger = logging.getLogger(__name__)

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> dict:
        use_stream = self.default_stream if stream is None else stream
        self.logger.debug("Calling Ollama generate: model=%s prompt_chars=%d", model, len(prompt))
        payload = {"model": model, "prompt": prompt, "stream": use_stream}
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=use_stream,
                timeout=self.timeout_sec,
            )
        except requests.RequestException as exc:
            raise OllamaError(f"Network error while calling Ollama: {exc}") from exc

        if response.status_code >= 400:
            raise OllamaError(f"Ollama generate failed: {response.status_code} {response.text}")
        if not use_stream:
            data = response.json()
            self.logger.debug("Ollama generate completed: model=%s eval_count=%s", model, data.get("eval_count"))
            return data

        full_text = ""
        final_obj: dict = {}
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                full_text += token
                if on_token:
                    on_token(token)
            if chunk.get("done"):
                final_obj = chunk
        if not final_obj:
            raise OllamaError("Streaming response did not contain a final done chunk.")
        final_obj["response"] = full_text
        self.logger.debug("Ollama stream completed: model=%s eval_count=%s", model, final_obj.get("eval_count"))
        return final_obj

    def ps(self) -> dict:
        self.logger.debug("Calling Ollama ps")
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=self.timeout_sec)
        except requests.RequestException as exc:
            raise OllamaError(f"Network error while calling Ollama ps: {exc}") from exc

        if response.status_code >= 400:
            raise OllamaError(f"Ollama ps failed: {response.status_code} {response.text}")

        return response.json()
