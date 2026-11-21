"""Central configuration (YAML files in configs/, env overrides for tests)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def data_dir() -> Path:
    return Path(os.environ.get("TRACEPILOT_DATA_DIR", ROOT / "data"))


def runtime_dir() -> Path:
    p = Path(os.environ.get("TRACEPILOT_RUNTIME_DIR", data_dir() / "runtime"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def documents_dir() -> Path:
    return Path(os.environ.get("TRACEPILOT_DOCS_DIR", data_dir() / "documents"))


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text()) or {}


def cfg(name: str) -> dict:
    d = dict(_load(name))
    if name == "models":
        d["llm"] = dict(d["llm"])
        if os.environ.get("TRACEPILOT_LLM_MODEL"):
            d["llm"]["model"] = os.environ["TRACEPILOT_LLM_MODEL"]
        if os.environ.get("OLLAMA_HOST_URL"):
            d["llm"]["host"] = os.environ["OLLAMA_HOST_URL"]
    return d
