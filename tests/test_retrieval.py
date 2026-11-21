import numpy as np

from src.embeddings.embedding_service import embed, embed_query
from src.knowledge.indexing import KnowledgeBase
from src.knowledge.retrieval import Retriever
from src.knowledge.vector_store import VectorStore


def test_embeddings_real_and_normalised():
    e = embed(["overfitting", "a completely unrelated sentence about cooking pasta"])
    assert e.shape == (2, 384) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)
    assert np.allclose(embed_query("overfitting"), e[0], atol=1e-5)              # deterministic
    s = embed(["dog", "puppy", "bridge"])
    assert s[0] @ s[1] > s[0] @ s[2]                                           # semantic, not random


def test_relevance_and_provenance(retriever):
    hits = retriever.search("What is overfitting?", 5)
    assert hits and hits[0].meta["doc_id"].startswith("ml_")
    h = hits[0]
    assert h.chunk_id.startswith(h.meta["doc_id"]) and h.meta["course_id"] == "ml" and h.score > 0.4


def test_course_isolation(retriever):
    for h in retriever.search("mean and median", 8, course_ids=["data"]):
        assert h.meta["course_id"] == "data"
    assert all(h.meta["course_id"] == "python" for h in retriever.search("overfitting regularization", 8, course_ids=["python"]))


def test_doc_and_type_filters(retriever):
    assert {h.meta["doc_id"] for h in retriever.search("price", 8, doc_ids=["product_prices"])} == {"product_prices"}
    assert {h.meta["file_type"] for h in retriever.search("price orders", 8, file_types=["csv"])} == {"csv"}


def test_unrelated_query_scores_low(retriever):
    assert max(h.score for h in retriever.search("capital city of France tourism", 5)) < 0.3


def test_csv_chunks_cite_row_ranges(retriever):
    h = next(x for x in retriever.search("regularization experiment run R05 L2 early stopping", 8, doc_ids=["ml_regularization_experiments"]) if x.meta.get("row_start"))
    assert h.location().startswith("rows ")


def test_neighbors_same_document(retriever):
    h = retriever.search("early stopping patience", 1)[0]
    nb = retriever.neighbors(h.chunk_id, 1)
    assert nb and {n.meta["doc_id"] for n in nb} == {h.meta["doc_id"]}


def test_persistence_after_restart(kb, tmp_path):
    n = kb.store.count()
    again = KnowledgeBase(VectorStore(kb.store.path), kb.db_path, kb.upload_dir)       # new client, same files
    assert again.store.count() == n and len(again.list_documents()) == len(kb.list_documents())
    assert Retriever(again).search("overfitting", 2)
