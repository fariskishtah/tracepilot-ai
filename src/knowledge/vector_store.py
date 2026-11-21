"""Persistent ChromaDB collection (cosine space) with metadata filtering."""
from __future__ import annotations

import threading
from pathlib import Path

import chromadb
import numpy as np

from src.config import cfg, runtime_dir

_lock = threading.Lock()


class VectorStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or runtime_dir() / "chroma")
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.col = self.client.get_or_create_collection(
            cfg("retrieval")["collection"], metadata={"hnsw:space": "cosine"})
        self.version = 0

    def upsert(self, ids: list[str], embeddings: np.ndarray, texts: list[str], metadatas: list[dict]) -> None:
        with _lock:
            for i in range(0, len(ids), 128):
                s = slice(i, i + 128)
                self.col.upsert(ids=ids[s], embeddings=embeddings[s].tolist(), documents=texts[s], metadatas=metadatas[s])
            self.version += 1

    def delete_doc(self, doc_id: str) -> None:
        with _lock:
            self.col.delete(where={"doc_id": doc_id})
            self.version += 1

    def count(self) -> int:
        return self.col.count()

    @staticmethod
    def build_where(course_ids=None, doc_ids=None, file_types=None) -> dict | None:
        conds = []
        if course_ids:
            conds.append({"course_id": {"$in": list(course_ids)}})
        if doc_ids:
            conds.append({"doc_id": {"$in": list(doc_ids)}})
        if file_types:
            conds.append({"file_type": {"$in": list(file_types)}})
        return None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}

    def query(self, emb: np.ndarray, n: int, where: dict | None) -> list[dict]:
        total = self.col.count()
        if total == 0:
            return []
        r = self.col.query(query_embeddings=[emb.tolist()], n_results=min(n, total), where=where,
                           include=["documents", "metadatas", "distances"])
        return [{"chunk_id": i, "text": d, "meta": m, "score": 1.0 - dist}
                for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def get(self, ids: list[str] | None = None, where: dict | None = None, with_emb: bool = False) -> list[dict]:
        inc = ["documents", "metadatas"] + (["embeddings"] if with_emb else [])
        r = self.col.get(ids=ids, where=where, include=inc)
        out = []
        for k, i in enumerate(r["ids"]):
            row = {"chunk_id": i, "text": r["documents"][k], "meta": r["metadatas"][k]}
            if with_emb:
                row["emb"] = np.asarray(r["embeddings"][k], dtype="float32")
            out.append(row)
        return out
