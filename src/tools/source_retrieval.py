from __future__ import annotations

from pydantic import BaseModel, Field

from src.tools.base import Tool, ToolContext, ToolError


class SourceInput(BaseModel):
    source_id: str = Field(pattern=r"^S\d+$", description="an already-retrieved source id, e.g. S2")
    window: int = Field(default=1, ge=0, le=2)


class RetrieveSourceContext(Tool):
    name = "retrieve_source_context"
    description = ("Fetch the neighbouring chunks (before/after) of an already retrieved source to read its full "
                   "surrounding context. Only works inside the selected course scope.")
    Input = SourceInput
    output_description = "chunks: [{source_id, location, text}] from the SAME document"
    examples = [{"source_id": "S1", "window": 1}]

    def run(self, a: SourceInput, ctx: ToolContext):
        it = ctx.evidence.get(a.source_id)
        if not it or it.kind != "document" or not it.chunk_id:
            raise ToolError(f"{a.source_id} is not a retrieved document source")
        if ctx.allowed_courses is not None and it.course_id not in ctx.allowed_courses:
            raise ToolError("source is outside the selected course scope")
        out, ids = [], []
        for h in ctx.retriever.neighbors(it.chunk_id, a.window):
            e = ctx.evidence.add_hit(h)
            ids.append(e.id)
            out.append({"source_id": e.id, "doc_id": e.doc_id, "location": e.location, "text": h.text[:800]})
        return {"around": a.source_id, "chunks": out}, ids, f"{len(out)} chunks around {a.source_id}"
