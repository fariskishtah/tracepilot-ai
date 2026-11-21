"""Hybrid retrieval: Chroma dense search + BM25, fused by reciprocal rank.

`score` is the dense cosine similarity (used for calibrated sufficiency thresholds).
Standard RAG and the agent share this retriever, so the comparison is fair.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import cfg
from src.embeddings.embedding_service import embed_query
from src.knowledge.indexing import KnowledgeBase

_tok = lambda s: re.findall(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*", s.lower())
STOP = set("the a an of and or to in is are was were be what how why which when does do for on with by as at from that this it its between about into than then".split())


@dataclass
class Hit:
    chunk_id: str
    text: str
    meta: dict
    score: float       # dense cosine
    fused: float

    def location(self) -> str:
        m = self.meta
        if m.get("page"):
            return f"p.{m['page']}"
        if m.get("row_start"):
            return f"rows {m['row_start']}-{m['row_end']}"
        return m.get("section") or "-"


class Retriever:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._bm25_ver = -1
        self._bm25 = None
        self._rows: list[dict] = []

    def _ensure_bm25(self) -> None:
        if self._bm25_ver == self.kb.store.version and self._bm25:
            return
        self._rows = self.kb.store.get()
        self._bm25 = BM25Okapi([[t for t in _tok(r["text"]) if t not in STOP] or ["_"] for r in self._rows]) if self._rows else None
        self._bm25_ver = self.kb.store.version

    def search(self, query: str, top_k: int | None = None, course_ids=None, doc_ids=None, file_types=None) -> list[Hit]:
        rc = cfg("retrieval")["retrieval"]
        top_k = min(top_k or rc["top_k"], rc["max_top_k"])
        where = self.kb.store.build_where(course_ids, doc_ids, file_types)
        qe = embed_query(query)
        dense = self.kb.store.query(qe, rc["dense_candidates"], where)
        self._ensure_bm25()
        allowed = lambda m: (not course_ids or m["course_id"] in course_ids) and (not doc_ids or m["doc_id"] in doc_ids) \
            and (not file_types or m["file_type"] in file_types)
        lex: list[tuple[str, float]] = []
        if self._bm25:
            qt = [t for t in _tok(query) if t not in STOP] or ["_"]
            sc = self._bm25.get_scores(qt)
            order = np.argsort(-sc)
            lex = [(self._rows[i]["chunk_id"], float(sc[i])) for i in order if sc[i] > 0 and allowed(self._rows[i]["meta"])][: rc["dense_candidates"]]
        K, w = 60.0, rc["lexical_weight"]
        fused: dict[str, float] = {}
        info = {d["chunk_id"]: d for d in dense}
        for r, d in enumerate(dense):
            fused[d["chunk_id"]] = fused.get(d["chunk_id"], 0) + (1 - w) / (K + r + 1)
        for r, (cid, _) in enumerate(lex):
            fused[cid] = fused.get(cid, 0) + w / (K + r + 1)
        missing = [c for c in fused if c not in info]
        if missing:  # lexical-only hits: compute true cosine from stored embeddings
            for row in self.kb.store.get(ids=missing, with_emb=True):
                info[row["chunk_id"]] = {"chunk_id": row["chunk_id"], "text": row["text"], "meta": row["meta"],
                                         "score": float(row["emb"] @ qe)}
        ranked = sorted(fused, key=lambda c: -fused[c])[:top_k]
        return [Hit(c, info[c]["text"], info[c]["meta"], info[c]["score"], fused[c]) for c in ranked]

    def neighbors(self, chunk_id: str, window: int = 1) -> list[Hit]:
        row = self.kb.store.get(ids=[chunk_id])
        if not row:
            return []
        m = row[0]["meta"]
        ids = [f"{m['doc_id']}:{i:04d}" for i in range(max(0, m["chunk_index"] - window), m["chunk_index"] + window + 1)]
        return [Hit(r["chunk_id"], r["text"], r["meta"], 0.0, 0.0) for r in sorted(self.kb.store.get(ids=ids), key=lambda r: r["meta"]["chunk_index"])]
