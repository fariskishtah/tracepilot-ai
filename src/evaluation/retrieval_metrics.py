from __future__ import annotations


def retrieved_docs(res: dict) -> list[str]:
    """Ordered unique doc ids of every document/dataset evidence item the system produced."""
    seen, out = set(), []
    for i, e in sorted(res["evidence"].items(), key=lambda kv: (kv[0][0], int(kv[0][1:]))):
        if e["kind"] in ("document", "dataset") and e.get("doc_id") and e["doc_id"] not in seen:
            seen.add(e["doc_id"])
            out.append(e["doc_id"])
    return out


def retrieval_scores(item: dict, res: dict, k: int = 5) -> dict:
    gold = set(item["gold_docs"])
    if not gold or not item["answerable"]:
        return {"coverage": None, "recall_at_k": None, "mrr": None}
    docs = retrieved_docs(res)
    cov = len(gold & set(docs)) / len(gold)
    rank = next((i + 1 for i, d in enumerate(docs) if d in gold), None)
    return {"coverage": cov, "recall_at_k": float(any(d in gold for d in docs[:k])), "mrr": 1.0 / rank if rank else 0.0}
