"""Real sentence embeddings (all-MiniLM-L6-v2), lazily loaded and L2-normalised."""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from src.config import cfg

_lock = threading.Lock()
_model = None


def _get():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            c = cfg("models")["embeddings"]
            _model = SentenceTransformer(c["model"], device=c["device"])
        return _model


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    return _get().encode(texts, batch_size=cfg("models")["embeddings"]["batch_size"],
                         normalize_embeddings=True, show_progress_bar=False).astype("float32")


@lru_cache(maxsize=2048)
def _embed_one(t: str) -> tuple:
    return tuple(embed([t])[0].tolist())


def embed_query(t: str) -> np.ndarray:
    return np.asarray(_embed_one(t), dtype="float32")


def model_name() -> str:
    return cfg("models")["embeddings"]["model"]
