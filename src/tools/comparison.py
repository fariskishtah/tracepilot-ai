"""Evidence comparison - fully extractive and deterministic: claims are verbatim source sentences, so the
comparison cannot invent content. Conflicts are preserved, never merged into a forced consensus."""
from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel, Field

from src.citations.validator import SENT, _words, parse_numbers
from src.embeddings.embedding_service import embed
from src.tools.base import Tool, ToolContext, ToolError

OUTDATED = re.compile(r"\b(superseded|deprecated|legacy|outdated|replaced|older|obsolete)\b|\b20[12]\d\s+edition\b", re.I)


class CompareInput(BaseModel):
    source_ids: list[str] = Field(min_length=2, max_length=8)
    aspects: list[str] = Field(default_factory=list, max_length=4,
                               description="what to compare, e.g. ['L2 regularization', 'early stopping']")


def _sents(t: str) -> list[str]:
    return [s.strip() for s in SENT.split(t) if len(s.strip()) > 25][:40]


class CompareEvidence(Tool):
    name = "compare_evidence"
    description = ("Compare retrieved sources: per-aspect statements, agreements, contradictions (numeric or "
                   "superseded/outdated sources) and aspects with missing evidence. Extractive, source-ids preserved.")
    Input = CompareInput
    output_description = "aspects: {aspect: [{source_id, sentence}]}, agreements, conflicts, missing_aspects"
    examples = [{"source_ids": ["S1", "S2"], "aspects": ["supervised learning", "unsupervised learning"]}]

    def run(self, a: CompareInput, ctx: ToolContext):
        srcs = []
        for sid in a.source_ids:
            it = ctx.evidence.get(sid)
            if not it or it.kind != "document":
                raise ToolError(f"{sid} is not a retrieved document source")
            srcs.append(it)
        rows = [(it.id, it.doc_id, s) for it in srcs for s in _sents(it.text)]
        if not rows:
            raise ToolError("no comparable sentences")
        E = embed([r[2] for r in rows] + list(a.aspects))
        SE, AE = E[:len(rows)], E[len(rows):]
        aspects, missing = {}, []
        for k, asp in enumerate(a.aspects):
            aw = _words(asp)
            sc = SE @ AE[k] * 0.75 + np.array([len(aw & _words(r[2])) / max(1, len(aw)) for r in rows]) * 0.25
            picked, per_src = [], {}
            for i in np.argsort(-sc):
                if sc[i] < 0.30:
                    break
                sid = rows[i][0]
                if per_src.get(sid, 0) >= 2 or len(picked) >= 4:
                    continue
                per_src[sid] = per_src.get(sid, 0) + 1
                picked.append({"source_id": sid, "sentence": rows[i][2], "relevance": round(float(sc[i]), 2)})
            if picked:
                aspects[asp] = picked
            else:
                missing.append(asp)
        agreements, conflicts = [], []
        n = len(rows)
        sim = SE @ SE.T
        for i in range(n):
            for j in range(i + 1, n):
                if rows[i][1] == rows[j][1] or sim[i, j] < 0.55:
                    continue
                ni = {round(v, 3) for v, _ in parse_numbers(rows[i][2])}
                nj = {round(v, 3) for v, _ in parse_numbers(rows[j][2])}
                pair = {"a": {"source_id": rows[i][0], "sentence": rows[i][2]}, "b": {"source_id": rows[j][0], "sentence": rows[j][2]}, "similarity": round(float(sim[i, j]), 2)}
                if (ni and nj and ni != nj) or bool(OUTDATED.search(rows[i][2])) != bool(OUTDATED.search(rows[j][2])):
                    pair["reason"] = "different numbers" if (ni and nj and ni != nj) else "one source is marked outdated/superseded"
                    conflicts.append(pair)
                elif sim[i, j] >= 0.75:
                    agreements.append(pair)
        data = {"aspects": aspects, "missing_aspects": missing, "agreements": agreements[:5], "conflicts": conflicts[:5],
                "sources": [{"source_id": s.id, "doc_id": s.doc_id, "title": s.title} for s in srcs]}
        return data, [], f"{len(aspects)} aspects covered, {len(missing)} missing, {len(agreements[:5])} agreements, {len(conflicts[:5])} conflicts"
