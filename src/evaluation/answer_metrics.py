"""Deterministic answer scoring against annotated acceptance criteria (no LLM judge)."""
from __future__ import annotations

import re

from src.citations.validator import parse_numbers, strip_cites

NOT_FOUND = re.compile(r"(could not find|cannot find|can't find|no (relevant |supporting )?(information|evidence|data|mention)|not (found|covered|contain|mention|provide|specif|available|present)|"
                       r"does not (contain|provide|mention|include|have)|insufficient|unable to (find|answer)|there is no|not (in|part of) the (knowledge|provided|selected))", re.I)


def numbers_match(answer: str, value: float, tol: float) -> bool:
    """Sign-agnostic: 'the difference between A and B' is a magnitude, and the direction is ambiguous in the question."""
    for v, _ in parse_numbers(strip_cites(answer)):
        for w in (v, -v):
            if abs(w - value) <= tol or (0 < abs(value) <= 1 and abs(w - value * 100) <= tol * 100 + 1e-9):
                return True
    return False


def score_answer(item: dict, res: dict) -> dict:
    ans = strip_cites(res["answer"])
    abstained = bool(res["abstained"]) or bool(NOT_FOUND.search(ans[:400]))
    groups = [any(re.search(p, ans, re.I | re.S) for p in ([g] if isinstance(g, str) else g)) for g in item["answer_checks"]]
    nums = [numbers_match(ans, n["value"], n["tol"]) for n in item["numeric_checks"]]
    forb_ok = not (item.get("forbidden_answer") and re.search(item["forbidden_answer"], ans, re.I))
    stale = item.get("stale")
    stale_ok = not (stale and re.search(stale["pattern"], ans, re.I) and not re.search(stale["ok_markers"], ans, re.I))
    if not item["answerable"]:
        correct = abstained
        text_ok = num_ok = True
    else:
        text_ok, num_ok = all(groups), all(nums)
        correct = (not res["abstained"]) and text_ok and num_ok and forb_ok and stale_ok
    return {"correct": bool(correct), "abstained": abstained, "text_ok": bool(text_ok), "numeric_ok": bool(num_ok) if item["numeric_checks"] else None,
            "forbidden_ok": forb_ok, "stale_ok": stale_ok, "n_text_groups": len(groups), "n_text_matched": sum(groups), "n_numeric": len(nums), "n_numeric_matched": sum(nums)}


def citation_scores(item: dict, res: dict) -> dict:
    gold = set(item["gold_docs"])
    cited_docs = [c["doc_id"] for c in res["citations"] if c.get("doc_id")]
    v = res.get("validation") or {}
    out = {"n_citations": len(res["citations"]), "support_rate": v.get("support_rate"), "n_cited_sentences": v.get("n_cited_sentences", 0),
           "unsupported": len(v.get("unsupported", [])), "uncited_numeric": len(v.get("uncited_numeric_sentences", [])),
           "invalid_ids": len(v.get("invalid_ids", []))}
    if gold and item["answerable"]:
        out["citation_hit"] = any(d in gold for d in cited_docs)
        out["citation_precision"] = (sum(1 for d in cited_docs if d in gold) / len(cited_docs)) if cited_docs else None
    else:
        out["citation_hit"] = None
        out["citation_precision"] = None
    return out
