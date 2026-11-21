"""Aggregation + paired bootstrap statistics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluation.agent_metrics import plan_completed, tool_scores
from src.evaluation.answer_metrics import citation_scores, score_answer
from src.evaluation.retrieval_metrics import retrieval_scores


def load_questions(split: str | None = None) -> list[dict]:
    from src.config import ROOT
    qs = [json.loads(l) for l in (ROOT / "data" / "evaluation" / "questions.jsonl").read_text().splitlines() if l.strip()]
    return [q for q in qs if split in (None, "all", q["split"])]


def score_item(item: dict, res: dict, agentic: bool = True) -> dict:
    a = score_answer(item, res)
    row = {"id": item["id"], "category": item["category"], "split": item["split"], "answerable": item["answerable"], "multi_step": item["multi_step"],
           **a, **{f"cit_{k}": v for k, v in citation_scores(item, res).items()}, **retrieval_scores(item, res),
           "latency_s": res["metrics"]["latency_s"], "tool_calls": res["metrics"]["tool_calls"], "llm_calls": res["metrics"]["llm_calls"],
           "tokens": res["metrics"]["prompt_tokens"] + res["metrics"]["completion_tokens"], "replans": res["metrics"]["replans"],
           "stop_reason": res["stop_reason"], "plan_source": res.get("plan_source")}
    ts = tool_scores(item, res)
    row.update({f"tool_{k}": v for k, v in ts.items()})
    row["plan_completed"] = plan_completed(res) if agentic else None
    if item["multi_step"] and item["answerable"]:
        has_gold = bool(item["gold_docs"])
        # answer-level completion (fair for every system): correct answer AND at least one gold document cited
        row["full_completion"] = bool(a["correct"] and (row["cit_citation_hit"] if has_gold and row["cit_citation_hit"] is not None else True))
        row["full_completion_with_path"] = bool(row["full_completion"] and ts["correct_path"])
        row["partial_completion"] = bool(a["n_text_matched"] + a["n_numeric_matched"] > 0 and not row["full_completion"])
    return row


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def frac(rows, key, cond=None):
    r = [x for x in rows if (cond is None or cond(x)) and x.get(key) is not None]
    return (sum(bool(x[key]) for x in r), len(r))


def paired_bootstrap(a: list[float], b: list[float], n: int = 5000, seed: int = 20261121) -> dict:
    """Paired bootstrap of mean(b - a) over questions (b = system B minus baseline a)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "b_only": int(((b == 1) & (a == 0)).sum()),
            "a_only": int(((a == 1) & (b == 0)).sum()), "significant": bool(lo > 0 or hi < 0)}


def latency_stats(rows):
    l = np.array([r["latency_s"] for r in rows])
    return {"mean": float(l.mean()), "median": float(np.median(l)), "p95": float(np.percentile(l, 95))} if len(l) else {}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float((c - m) / d), float((c + m) / d)]
