"""Secure document loaders for PDF / DOCX / CSV / TXT.

Text is extracted only; macros, embedded scripts and links are never executed or followed.
Every Segment carries the real location it came from (page, section, CSV row range).
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import cfg
from src.ingestion.cleaning import clean_text, strip_repeated_lines

ALLOWED = {".pdf", ".docx", ".csv", ".txt"}
MAGIC = {".pdf": b"%PDF", ".docx": b"PK"}


class IngestionError(ValueError):
    pass


@dataclass
class Segment:
    text: str
    page: int | None = None
    section: str | None = None
    row_start: int | None = None      # 1-based CSV data-row index
    row_end: int | None = None
    columns: list[str] | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class LoadedDocument:
    filename: str
    file_type: str
    sha256: str
    title: str
    segments: list[Segment]
    n_pages: int | None = None
    n_rows: int | None = None
    columns: list[str] | None = None


def validate_file(path: Path, max_bytes: int | None = None) -> bytes:
    max_bytes = max_bytes or cfg("app")["upload"]["max_bytes"]
    p = Path(path)
    if p.suffix.lower() not in ALLOWED:
        raise IngestionError(f"unsupported file type: {p.suffix!r}")
    if p.is_symlink() or not p.is_file():
        raise IngestionError("not a regular file")
    size = p.stat().st_size
    if size == 0:
        raise IngestionError("empty file")
    if size > max_bytes:
        raise IngestionError(f"file too large ({size} > {max_bytes} bytes)")
    data = p.read_bytes()
    magic = MAGIC.get(p.suffix.lower())
    if magic and not data.startswith(magic):
        raise IngestionError("file content does not match its extension")
    return data


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _title_from(name: str) -> str:
    return Path(name).stem.replace("_", " ").strip().title()


def _load_pdf(data: bytes, name: str) -> LoadedDocument:
    import pymupdf
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise IngestionError(f"unreadable PDF: {type(e).__name__}") from e
    pages = [clean_text(p.get_text()) for p in doc]
    pages = strip_repeated_lines(pages)
    segs = []
    for i, t in enumerate(pages, 1):
        t = clean_text(t)
        if not t:
            continue
        m = re.match(r"^(\d+\.\s+[^\n]{3,80})", t)
        segs.append(Segment(t, page=i, section=m.group(1) if m else None))
    return LoadedDocument(name, "pdf", sha256_bytes(data), _title_from(name), segs, n_pages=len(pages))


def _load_docx(data: bytes, name: str) -> LoadedDocument:
    import docx
    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as e:
        raise IngestionError(f"unreadable DOCX: {type(e).__name__}") from e
    segs, cur, buf = [], None, []

    def flush():
        t = clean_text("\n".join(buf))
        if t:
            segs.append(Segment(t, section=cur))
        buf.clear()

    for p in d.paragraphs:
        if p.style is not None and p.style.name.lower().startswith(("heading", "title")) and p.text.strip():
            flush()
            cur = p.text.strip()
            buf.append(cur)
        elif p.text.strip():
            buf.append(p.text)
    flush()
    return LoadedDocument(name, "docx", sha256_bytes(data), _title_from(name), segs)


def _load_txt(data: bytes, name: str) -> LoadedDocument:
    text = clean_text(data.decode("utf-8", errors="replace"))
    segs, cur = [], None
    for block in re.split(r"\n\s*\n", text):
        b = block.strip()
        if not b or set(b) <= set("=-_ "):
            continue
        first = b.splitlines()[0]
        if len(first) < 90 and (first.isupper() or first.endswith(":")):
            cur = first.strip(":")
        segs.append(Segment(b, section=cur))
    return LoadedDocument(name, "txt", sha256_bytes(data), _title_from(name), segs)


def _load_csv(data: bytes, name: str) -> LoadedDocument:
    try:
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace"))))
    except csv.Error as e:
        raise IngestionError(f"unreadable CSV: {e}") from e
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise IngestionError("CSV needs a header and at least one data row")
    header, body = [h.strip() for h in rows[0]], rows[1:]
    n = cfg("retrieval")["chunking"]["csv_rows_per_chunk"]
    segs = [Segment(
        f"Dataset {Path(name).stem}: {len(body)} rows, columns: {', '.join(header)}.",
        section="dataset overview", row_start=0, row_end=0, columns=header)]
    for i in range(0, len(body), n):
        part = body[i:i + n]
        lines = ["; ".join(f"{h}={v.strip()}" for h, v in zip(header, r)) for r in part]
        segs.append(Segment("\n".join(f"row {i + j + 1}: {l}" for j, l in enumerate(lines)),
                            section="data rows", row_start=i + 1, row_end=i + len(part), columns=header))
    return LoadedDocument(name, "csv", sha256_bytes(data), _title_from(name), segs, n_rows=len(body), columns=header)


def load_document(path: Path) -> LoadedDocument:
    data = validate_file(path)
    ext = Path(path).suffix.lower()
    name = Path(path).name
    return {".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt, ".csv": _load_csv}[ext](data, name)
