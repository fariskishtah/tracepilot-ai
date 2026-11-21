from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.tools.base import Tool, ToolContext, ToolError


class SearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    course_ids: list[str] | None = Field(default=None, max_length=6)
    doc_ids: list[str] | None = Field(default=None, max_length=10)
    file_types: list[Literal["pdf", "docx", "csv", "txt"]] | None = None
    top_k: int = Field(default=5, ge=1, le=8)


def scope_courses(ctx: ToolContext, requested: list[str] | None) -> list[str] | None:
    """The user-selected scope always wins: a tool can narrow it but never widen it (course isolation)."""
    if ctx.allowed_courses is None:
        return requested
    if not requested:
        return ctx.allowed_courses
    inter = [c for c in requested if c in ctx.allowed_courses]
    return inter or ctx.allowed_courses


class SearchKnowledgeBase(Tool):
    name = "search_knowledge_base"
    description = ("Semantic + lexical search over the indexed knowledge base. Use for any question that needs facts, "
                   "definitions, procedures or dataset descriptions from the documents.")
    Input = SearchInput
    output_description = "hits: [{source_id, doc_id, title, course_id, location, score, text}] (source ids S1.. are citable)"
    examples = [{"query": "difference between supervised and unsupervised learning", "course_ids": ["ml"], "top_k": 5}]

    def run(self, a: SearchInput, ctx: ToolContext):
        courses = scope_courses(ctx, a.course_ids)
        hits = ctx.retriever.search(a.query, a.top_k, courses, a.doc_ids, a.file_types)
        out, ids = [], []
        for h in hits:
            it = ctx.evidence.add_hit(h)
            ids.append(it.id)
            out.append({"source_id": it.id, "doc_id": it.doc_id, "title": it.title, "course_id": it.course_id,
                        "location": it.location, "score": round(h.score, 3), "text": h.text[:700]})
        best = max((h.score for h in hits), default=0.0)
        return {"query": a.query, "hits": out, "best_score": round(best, 3)}, ids, f"{len(out)} chunks, best cosine {best:.2f}"
