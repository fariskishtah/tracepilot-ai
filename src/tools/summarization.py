"""Summarize evidence. LLM-written bullets are accepted only if the citation validator confirms each bullet
is supported by the cited source; otherwise an extractive fallback is used. Summaries are never primary evidence."""
from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel, Field

from src.citations.validator import CITE, SENT, ids_in, sentence_supported, strip_cites
from src.embeddings.embedding_service import embed
from src.llm import LLMError
from src.security.guard import wrap_untrusted, SYSTEM_SAFETY
from src.tools.base import Tool, ToolContext, ToolError


class SummarizeInput(BaseModel):
    source_ids: list[str] = Field(min_length=1, max_length=6)
    focus: str | None = Field(default=None, max_length=200)
    max_bullets: int = Field(default=4, ge=1, le=6)


class SummarizeEvidence(Tool):
    name = "summarize_evidence"
    description = "Summarize specific retrieved sources (optionally around a focus topic), keeping source ids. Uses only the supplied evidence."
    Input = SummarizeInput
    output_description = "summary bullets [{text, source_ids}] + method (llm|extractive); generated evidence id G#"
    examples = [{"source_ids": ["S1", "S3"], "focus": "advantages", "max_bullets": 3}]
    uses_llm = True

    def run(self, a: SummarizeInput, ctx: ToolContext):
        items = []
        for sid in a.source_ids:
            it = ctx.evidence.get(sid)
            if not it or it.kind not in ("document", "dataset", "calculation"):
                raise ToolError(f"{sid} is not a primary source")
            items.append(it)
        method, bullets = "llm", []
        try:
            ev = "\n".join(f"[{i.id}] ({i.title}) {i.text[:900]}" for i in items)
            user = (f"Focus: {a.focus or 'the main points'}\n\n{wrap_untrusted(ev)}\n\n"
                    f"Write {a.max_bullets} or fewer bullet points. Each bullet is ONE sentence that ends with the source id in brackets, e.g. [S1]. "
                    "Use only facts present in the evidence.")
            raw = ctx.llm.chat(SYSTEM_SAFETY + " You summarize supplied evidence only.", user, max_tokens=260)
            for line in raw.splitlines():
                line = line.strip(" -*•\t")
                cited = [c for c in ids_in(line) if c in a.source_ids]
                if len(strip_cites(line)) < 15 or not cited:
                    continue
                ok, _, _ = sentence_supported(line, [ctx.evidence.items[c].text for c in cited])
                if ok:
                    bullets.append({"text": strip_cites(line), "source_ids": sorted(set(cited))})
        except LLMError:
            pass
        if not bullets:  # extractive fallback: most central / focus-relevant verbatim sentences
            method = "extractive"
            rows = [(i.id, s.strip()) for i in items for s in SENT.split(i.text) if len(s.strip()) > 30]
            if not rows:
                raise ToolError("nothing to summarize")
            E = embed([r[1] for r in rows] + [a.focus or " ".join(r[1] for r in rows)[:500]])
            sc = E[:-1] @ E[-1]
            for k in np.argsort(-sc)[: a.max_bullets]:
                bullets.append({"text": rows[k][1], "source_ids": [rows[k][0]]})
        bullets = bullets[: a.max_bullets]
        used = sorted({s for b in bullets for s in b["source_ids"]})
        text = " ".join(f"{b['text']} [{','.join(b['source_ids'])}]" for b in bullets)
        g = ctx.evidence.add_summary(text, used)
        return {"bullets": bullets, "method": method, "derived_from": used}, [g.id], f"{len(bullets)} bullets ({method}) from {', '.join(used)}"
