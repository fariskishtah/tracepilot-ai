"""Sentence-aware chunking with overlap. Chunk metadata is inherited from its Segment."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from src.config import cfg
from src.ingestion.loaders import LoadedDocument, Segment


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    course_id: str
    title: str
    filename: str
    file_type: str
    text: str
    page: int | None
    section: str | None
    row_start: int | None
    row_end: int | None
    chunk_index: int
    content_hash: str

    def metadata(self) -> dict:
        """Chroma metadata (no None values allowed)."""
        d = {k: v for k, v in self.__dict__.items() if k != "text" and v is not None}
        return d


def _split(text: str, target: int, overlap: int) -> list[str]:
    if len(text) <= target:
        return [text]
    sents = re.split(r"(?<=[.!?])\s+|\n+", text)
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > target:
            out.append(cur.strip())
            cur = cur[-overlap:].split(" ", 1)[-1] + " " if overlap else ""
        cur += s + " "
    if cur.strip():
        out.append(cur.strip())
    return out


def chunk_document(doc: LoadedDocument, doc_id: str, course_id: str) -> list[Chunk]:
    c = cfg("retrieval")["chunking"]
    chunks: list[Chunk] = []
    for seg in doc.segments:
        parts = [seg.text] if doc.file_type == "csv" else _split(seg.text, c["target_chars"], c["overlap_chars"])
        for p in parts:
            if len(p.strip()) < 20 and doc.file_type != "csv":
                continue
            i = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}:{i:04d}", doc_id=doc_id, course_id=course_id, title=doc.title,
                filename=doc.filename, file_type=doc.file_type, text=p.strip(), page=seg.page,
                section=seg.section, row_start=seg.row_start, row_end=seg.row_end, chunk_index=i,
                content_hash=hashlib.sha256(p.strip().encode()).hexdigest()[:16]))
    return chunks
