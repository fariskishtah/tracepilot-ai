import shutil
from pathlib import Path

import pytest

from src.ingestion.chunking import chunk_document
from src.ingestion.loaders import IngestionError, load_document, validate_file
from src.knowledge.indexing import KnowledgeBase, safe_course, safe_name
from src.knowledge.vector_store import VectorStore

D = Path(__file__).resolve().parents[1] / "data" / "documents"


def test_pdf_pages_and_sections():
    d = load_document(D / "ml/ml_lecture_notes.pdf")
    assert d.file_type == "pdf" and d.n_pages >= 6
    assert all(s.page for s in d.segments)
    assert "CourseAtlas Demo Course Pack" not in " ".join(s.text for s in d.segments)   # repeated header stripped
    assert any("Overfitting" in s.text for s in d.segments)


def test_docx_sections():
    d = load_document(D / "ml/ml_project_handbook.docx")
    assert d.file_type == "docx" and any(s.section for s in d.segments)


def test_txt_and_csv_metadata():
    t = load_document(D / "statistics/statistics_robust_summaries.txt")
    assert t.file_type == "txt" and t.segments
    c = load_document(D / "data/product_prices.csv")
    assert c.n_rows == 40 and c.columns[:2] == ["product_id", "category"]
    assert c.segments[0].section == "dataset overview"
    assert c.segments[1].row_start == 1 and c.segments[1].row_end == 8      # real row references


def test_chunks_have_provenance():
    d = load_document(D / "ml/ml_lecture_notes.pdf")
    ch = chunk_document(d, "ml_lecture_notes", "ml")
    assert ch and len({c.chunk_id for c in ch}) == len(ch)
    assert all(c.doc_id == "ml_lecture_notes" and c.course_id == "ml" and len(c.content_hash) == 16 for c in ch)
    assert all(c.page for c in ch)


def test_rejects_bad_files(tmp_path):
    (tmp_path / "x.exe").write_bytes(b"MZ")
    with pytest.raises(IngestionError):
        validate_file(tmp_path / "x.exe")
    (tmp_path / "fake.pdf").write_bytes(b"not a pdf at all")
    with pytest.raises(IngestionError):
        load_document(tmp_path / "fake.pdf")
    (tmp_path / "empty.txt").write_bytes(b"")
    with pytest.raises(IngestionError):
        validate_file(tmp_path / "empty.txt")
    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * 100)
    with pytest.raises(IngestionError):
        validate_file(big, max_bytes=10)
    (tmp_path / "m.docm").write_bytes(b"PK")
    with pytest.raises(IngestionError):                    # macro-enabled documents are not an allowed type
        validate_file(tmp_path / "m.docm")


def test_symlink_rejected(tmp_path):
    (tmp_path / "a.txt").write_text("hello world " * 10)
    (tmp_path / "l.txt").symlink_to(tmp_path / "a.txt")
    with pytest.raises(IngestionError):
        validate_file(tmp_path / "l.txt")


def test_safe_names():
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("a b/c d.txt") == "c_d.txt"
    assert safe_course("../Evil Course!") == "evil_course"


def fresh_kb(tmp_path):
    return KnowledgeBase(VectorStore(tmp_path / "c"), tmp_path / "r.sqlite", tmp_path / "u")


def test_duplicate_update_delete_cycle(tmp_path):
    kb = fresh_kb(tmp_path)
    f = tmp_path / "notes.txt"
    f.write_text("Gradient descent updates parameters step by step. " * 12)
    assert kb.ingest_file(f, "ml", store_copy=True).status == "indexed"
    n = kb.store.count()
    assert kb.ingest_file(f, "ml").status == "duplicate" and kb.store.count() == n
    f.write_text("Completely different content about trees and forests. " * 12)
    assert kb.ingest_file(f, "ml").status == "updated"
    assert "different" in " ".join(r["text"] for r in kb.store.get(where={"doc_id": "notes"}))
    assert kb.reindex_document("notes").status in ("indexed", "updated")
    assert kb.delete_document("notes") and kb.store.count() == 0 and not kb.list_documents()


def test_same_content_other_name_is_duplicate(tmp_path):
    kb = fresh_kb(tmp_path)
    (tmp_path / "a.txt").write_text("Alpha beta gamma delta epsilon. " * 10)
    (tmp_path / "b.txt").write_text("Alpha beta gamma delta epsilon. " * 10)
    kb.ingest_file(tmp_path / "a.txt", "c1")
    r = kb.ingest_file(tmp_path / "b.txt", "c1")
    assert r.status == "duplicate" and r.doc_id == "a"


def test_doc_id_cannot_cross_courses(tmp_path):
    kb = fresh_kb(tmp_path)
    (tmp_path / "a.txt").write_text("Alpha beta gamma delta epsilon. " * 10)
    kb.ingest_file(tmp_path / "a.txt", "c1")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "a.txt").write_text("Different words entirely here. " * 10)
    with pytest.raises(IngestionError):
        kb.ingest_file(tmp_path / "other" / "a.txt", "c2")
