"""Collect REAL failure cases (by failure type) from result files. Usage: python scripts/failure_analysis.py --split test --system full"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.benchmarking import load_questions  # noqa: E402

M = ROOT / "artifacts" / "metrics"
TYPES = {
    "wrong_or_missing_tool": lambda r, q: not r.get("tool_required_ok", True) or r.get("tool_forbidden_used"),
    "unnecessary_tool_calls": lambda r, q: r.get("tool_unnecessary_calls", 0) > 0,
    "failed_retrieval": lambda r, q: q["answerable"] and q["gold_docs"] and (r.get("coverage") or 0) == 0,
    "incomplete_multistep": lambda r, q: q["multi_step"] and q["answerable"] and not r.get("full_completion", True),
    "unsupported_or_uncited_claims": lambda r, q: (r.get("cit_unsupported") or 0) > 0 or (r.get("cit_uncited_numeric") or 0) > 0,
    "numerical_error": lambda r, q: r.get("numeric_ok") is False,
    "false_abstention": lambda r, q: q["answerable"] and r["abstained"],
    "answered_unanswerable": lambda r, q: not q["answerable"] and not r["abstained"],
    "injection_followed": lambda r, q: r.get("forbidden_ok") is False,
    "stale_source_used": lambda r, q: r.get("stale_ok") is False,
    "limit_exceeded": lambda r, q: r.get("stop_reason") in ("tool_budget_exhausted", "timeout"),
    "planner_fallback": lambda r, q: r.get("plan_source") == "rules_fallback",
    "wrong_answer_other": lambda r, q: q["answerable"] and not r["correct"] and not r["abstained"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--system", default="full")
    ap.add_argument("--max", type=int, default=3)
    a = ap.parse_args()
    qs = {q["id"]: q for q in load_questions(a.split)}
    rows = [json.loads(l) for l in (M / f"{a.split}_{a.system}.jsonl").read_text().splitlines()]
    out = {}
    for t, f in TYPES.items():
        hits = [r for r in rows if f(r, qs[r["id"]])]
        out[t] = {"count": len(hits), "of": len(rows), "examples": [
            {"id": r["id"], "category": r["category"], "question": qs[r["id"]]["question"], "gold": qs[r["id"]]["gold_answer"], "answer": r["answer"][:420],
             "tools": r["tool_sequence"], "stop_reason": r["stop_reason"], "correct": r["correct"]} for r in hits[: a.max]]}
        print(f"{t}: {len(hits)}/{len(rows)}")
    (M / f"failures_{a.split}_{a.system}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
