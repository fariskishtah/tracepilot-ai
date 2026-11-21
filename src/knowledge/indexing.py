"""Document registry + indexing: validation, dedup, update, delete, reindex, isolation."""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import documents_dir, runtime_dir
from src.embeddings.embedding_service import embed
from src.ingestion.chunking import chunk_document
from src.ingestion.loaders import IngestionError, load_document
from src.knowledge.vector_store import VectorStore

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    n = _SAFE.sub("_", Path(name).name).strip("._")
    if not n:
        raise IngestionError("invalid file name")
    return n


def safe_course(c: str) -> str:
    c = _SAFE.sub("_", c.strip().lower()).strip("._")
    if not c:
        raise IngestionError("invalid course id")
    return c


@dataclass
class IngestResult:
    doc_id: str
    status: str            # indexed | duplicate | updated
    n_chunks: int


class KnowledgeBase:
    def __init__(self, store: VectorStore | None = None, registry_path: Path | None = None, upload_dir: Path | None = None):
        rt = runtime_dir()
        self.store = store or VectorStore()
        self.upload_dir = Path(upload_dir or rt / "uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(registry_path or rt / "registry.sqlite")
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS documents(
            doc_id TEXT PRIMARY KEY, course_id TEXT, filename TEXT, file_type TEXT, sha256 TEXT, title TEXT,
            n_chunks INT, n_pages INT, n_rows INT, columns TEXT, origin TEXT, path TEXT, status TEXT, indexed_at REAL)""")
        self.db.commit()

    # ---- queries
    def list_documents(self, course_id: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM documents", ()
        if course_id:
            q, a = q + " WHERE course_id=?", (course_id,)
        cur = self.db.execute(q + " ORDER BY course_id, filename", a)
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
        for r in rows:
            r["columns"] = json.loads(r["columns"]) if r["columns"] else None
        return rows

    def get_document(self, doc_id: str) -> dict | None:
        return next((d for d in self.list_documents() if d["doc_id"] == doc_id), None)

    def courses(self) -> list[str]:
        return sorted({d["course_id"] for d in self.list_documents()})

    def dataset_catalog(self, course_ids=None) -> list[dict]:
        out = []
        for d in self.list_documents():
            if d["file_type"] == "csv" and (not course_ids or d["course_id"] in course_ids):
                out.append({"dataset_id": d["doc_id"], "course_id": d["course_id"], "columns": d["columns"], "n_rows": d["n_rows"]})
        return out

    # ---- mutations
    def ingest_file(self, path: Path, course_id: str, origin: str = "upload", store_copy: bool = False) -> IngestResult:
        course_id = safe_course(course_id)
        path = Path(path)
        doc = load_document(path)                       # validation + extraction
        doc_id = safe_name(path.stem)
        ex = self.get_document(doc_id)
        if ex and ex["course_id"] != course_id:
            raise IngestionError(f"document id '{doc_id}' already exists in course '{ex['course_id']}'")
        if ex and ex["sha256"] == doc.sha256 and ex["status"] == "indexed":
            return IngestResult(doc_id, "duplicate", ex["n_chunks"])
        for d in self.list_documents():                 # same content under another name
            if d["sha256"] == doc.sha256 and d["course_id"] == course_id and d["doc_id"] != doc_id:
                return IngestResult(d["doc_id"], "duplicate", d["n_chunks"])
        chunks = chunk_document(doc, doc_id, course_id)
        if not chunks:
            raise IngestionError("no extractable text")
        stored = path
        if store_copy:
            dest = self.upload_dir / course_id / safe_name(path.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            stored = dest
        status = "updated" if ex else "indexed"
        if ex:
            self.store.delete_doc(doc_id)
        self.store.upsert([c.chunk_id for c in chunks], embed([c.text for c in chunks]),
                          [c.text for c in chunks], [c.metadata() for c in chunks])
        self.db.execute("INSERT OR REPLACE INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (doc_id, course_id, doc.filename, doc.file_type, doc.sha256, doc.title, len(chunks),
                         doc.n_pages, doc.n_rows, json.dumps(doc.columns) if doc.columns else None,
                         origin, str(stored), "indexed", time.time()))
        self.db.commit()
        return IngestResult(doc_id, status, len(chunks))

    def delete_document(self, doc_id: str) -> bool:
        d = self.get_document(doc_id)
        if not d:
            return False
        self.store.delete_doc(doc_id)
        self.db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        self.db.commit()
        if d["origin"] == "upload":
            p = Path(d["path"])
            if self.upload_dir in p.resolve().parents and p.exists():
                p.unlink()
        return True

    def reindex_document(self, doc_id: str) -> IngestResult:
        d = self.get_document(doc_id)
        if not d:
            raise IngestionError("unknown document")
        self.db.execute("UPDATE documents SET sha256='' WHERE doc_id=?", (doc_id,))
        self.db.commit()
        return self.ingest_file(Path(d["path"]), d["course_id"], d["origin"])

    def index_directory(self, root: Path | None = None) -> list[IngestResult]:
        """Index the packaged base corpus: <root>/<course_id>/<files>."""
        root = Path(root or documents_dir())
        res = []
        for p in sorted(root.glob("*/*")):
            if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".csv", ".txt"}:
                res.append(self.ingest_file(p, p.parent.name, origin="base"))
        return res

    def dataset_path(self, dataset_id: str, course_ids=None) -> Path | None:
        d = self.get_document(dataset_id)
        if not d or d["file_type"] != "csv" or (course_ids and d["course_id"] not in course_ids):
            return None
        return Path(d["path"])
