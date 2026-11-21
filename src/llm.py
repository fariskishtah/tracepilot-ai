"""Local LLM client (Ollama HTTP API). No paid services.

Structured output uses Ollama's JSON-schema `format` constraint, and every result is
re-validated with Pydantic. Invalid JSON is never treated as a successful tool call.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Type

import requests
from pydantic import BaseModel, ValidationError

from src.config import cfg


class LLMError(RuntimeError):
    pass


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, o: "LLMUsage") -> None:
        self.calls += o.calls
        self.prompt_tokens += o.prompt_tokens
        self.completion_tokens += o.completion_tokens
        self.seconds += o.seconds


@dataclass
class OllamaLLM:
    model: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)

    def __post_init__(self) -> None:
        c = cfg("models")["llm"]
        self.c = c
        self.model = self.model or c["model"]

    def _post(self, payload: dict, timeout: float | None = None) -> dict:
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.c['host']}/api/chat", json=payload, timeout=timeout or self.c["timeout_s"])
            r.raise_for_status()
        except requests.RequestException as e:  # connection refused, timeout, 5xx
            raise LLMError(f"ollama request failed: {type(e).__name__}") from e
        d = r.json()
        self.usage.calls += 1
        self.usage.prompt_tokens += d.get("prompt_eval_count", 0)
        self.usage.completion_tokens += d.get("eval_count", 0)
        self.usage.seconds += time.perf_counter() - t0
        return d

    def _payload(self, messages: list[dict], max_tokens: int, fmt: Any = None) -> dict:
        p = {
            "model": self.model, "messages": messages, "stream": False,
            "keep_alive": self.c["keep_alive"],
            "options": {"temperature": self.c["temperature"], "seed": self.c["seed"],
                        "num_ctx": self.c["num_ctx"], "num_predict": max_tokens},
        }
        if fmt is not None:
            p["format"] = fmt
        return p

    def chat(self, system: str, user: str, max_tokens: int | None = None) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        d = self._post(self._payload(msgs, max_tokens or self.c["num_predict_answer"]))
        return d["message"]["content"].strip()

    def structured(self, system: str, user: str, schema_model: Type[BaseModel], max_tokens: int | None = None, retry_seed: bool = False) -> BaseModel:
        """Schema-constrained generation + strict Pydantic validation. Raises LLMError if invalid."""
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        pl = self._payload(msgs, max_tokens or self.c["num_predict_plan"], schema_model.model_json_schema())
        if retry_seed:
            pl["options"].update(seed=self.c["seed"] + 101, temperature=0.3, repeat_penalty=1.1)
        d = self._post(pl)
        raw = d["message"]["content"]
        try:
            return schema_model.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            raise LLMError(f"invalid structured output: {type(e).__name__}: {str(e)[:200]}") from e

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.c['host']}/api/tags", timeout=3)
            return any(m["name"] == self.model for m in r.json().get("models", []))
        except Exception:
            return False
