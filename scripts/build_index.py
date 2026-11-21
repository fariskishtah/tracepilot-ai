"""Index the packaged corpus (data/documents) into the persistent Chroma store + registry."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.knowledge.indexing import KnowledgeBase  # noqa: E402

if __name__ == "__main__":
    kb = KnowledgeBase()
    res = kb.index_directory()
    print({r.status: sum(1 for x in res if x.status == r.status) for r in res}, "chunks:", kb.store.count(), "courses:", kb.courses())
