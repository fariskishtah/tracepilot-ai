"""Summarise result files into tables + paired statistics. Usage:
   python scripts/summarize.py --split dev --systems standard full [--baseline standard] [--out summary_dev.json]"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.benchmarking import frac, latency_stats, mean, paired_bootstrap, wilson  # noqa: E402

M = ROOT / "artifacts" / "metrics"


def load(split, system, tag=""):
    p = M / f"{split}_{system}{tag}.jsonl"
    rows = {r["id"]: r for r in map(json.loads, p.read_text().splitlines())} if p.exists() else {}
    for r in rows.values():       # derive completion flags from stored fields (definition is identical for all systems)
        if r["multi_step"] and r["answerable"]:
            hit = r.get("cit_citation_hit")
            r["full_completion"] = bool(r["correct"] and (hit if hit is not None else True))
            r["full_completion_with_path"] = bool(r["full_completion"] and r["tool_correct_path"])
    return rows


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "n/a"


def summarize(rows: list[dict], agentic: bool) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    ms = [r for r in ans if r["multi_step"]]
    num = [r for r in rows if r.get("numeric_ok") is not None]
    out = {
        "n": len(rows),
        "answer_correctness (all)": frac(rows, "correct"), "answer_correctness (answerable)": frac(ans, "correct"),
        "correct_abstention (unanswerable)": frac(una, "correct"),
        "false_abstention (answerable)": (sum(1 for r in ans if r["abstained"]), len(ans)),
        "numerical_accuracy": frac(num, "numeric_ok"),
        "multi_step_full_completion": frac(ms, "full_completion"), "multi_step_full_completion_with_correct_tool_path": frac(ms, "full_completion_with_path"),
        "citation_hit (answerable w/ gold docs)": frac(rows, "cit_citation_hit"),
        "mean_citation_support_rate": mean([r["cit_support_rate"] for r in rows if r.get("cit_support_rate") is not None]),
        "unsupported_cited_sentences": (sum(r["cit_unsupported"] for r in rows), sum(r["cit_n_cited_sentences"] for r in rows)),
        "retrieval_coverage": mean([r["coverage"] for r in rows]), "recall@5": mean([r["recall_at_k"] for r in rows]), "MRR": mean([r["mrr"] for r in rows]),
        "latency": latency_stats(rows), "mean_tool_calls": mean([r["tool_calls"] for r in rows]), "mean_llm_calls": mean([r["llm_calls"] for r in rows]),
        "mean_tokens": mean([r["tokens"] for r in rows]),
    }
    if agentic:
        out.update({
            "tool_correct_path": frac(rows, "tool_correct_path"), "tool_required_ok(recall)": frac(rows, "tool_required_ok"),
            "mean_tool_precision": mean([r["tool_tool_precision"] for r in rows]),
            "forbidden_tool_used": frac(rows, "tool_forbidden_used"),
            "unnecessary_call_rate": (sum(r["tool_unnecessary_calls"] for r in rows), sum(r["tool_n_calls"] for r in rows)),
            "plan_completion": frac(rows, "plan_completed"), "replanned_requests": (sum(1 for r in rows if r["replans"]), len(rows)),
            "tool_failures_per_req": None, "planner_rules_fallback": (sum(1 for r in rows if r.get("plan_source") == "rules_fallback"), len(rows)),
        })
    return out


def by_category(rows):
    cats = sorted({r["category"] for r in rows})
    return {c: frac([r for r in rows if r["category"] == c], "correct") for c in cats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--systems", nargs="+", default=["standard", "full"])
    ap.add_argument("--baseline", default="standard")
    ap.add_argument("--tag", default="")
    ap.add_argument("--std-tag", default=None, help="tag for the standard-RAG file (defaults to --tag)")
    ap.add_argument("--out")
    a = ap.parse_args()
    data = {s: load(a.split, s, (a.std_tag if a.std_tag is not None else a.tag) if s == "standard" else a.tag) for s in a.systems}
    ids = sorted(set.intersection(*[set(d) for d in data.values() if d])) if any(data.values()) else []
    res = {"split": a.split, "n_common": len(ids), "systems": {}, "by_category": {}, "paired_vs_baseline": {}}
    for s, d in data.items():
        rows = [d[i] for i in ids]
        res["systems"][s] = summarize(rows, s != "standard")
        res["by_category"][s] = by_category(rows)
    if a.baseline in data:
        base = [data[a.baseline][i]["correct"] * 1.0 for i in ids]
        for s, d in data.items():
            if s == a.baseline:
                continue
            res["paired_vs_baseline"][s] = {"overall": paired_bootstrap(base, [d[i]["correct"] * 1.0 for i in ids])}
            cats = sorted({d[i]["category"] for i in ids})
            for c in cats:
                ci = [i for i in ids if d[i]["category"] == c]
                res["paired_vs_baseline"][s][c] = paired_bootstrap([data[a.baseline][i]["correct"] * 1.0 for i in ci], [d[i]["correct"] * 1.0 for i in ci], n=2000)
    # flat table for the UI/README
    keys = ["answer_correctness (all)", "answer_correctness (answerable)", "correct_abstention (unanswerable)", "numerical_accuracy", "multi_step_full_completion"]
    res["table"] = [{"metric": k, **{s: pct(*res["systems"][s][k]) for s in data}} for k in keys]
    res["table"] += [{"metric": "mean latency (s)", **{s: f"{res['systems'][s]['latency'].get('mean', 0):.1f}" for s in data}},
                     {"metric": "p95 latency (s)", **{s: f"{res['systems'][s]['latency'].get('p95', 0):.1f}" for s in data}},
                     {"metric": "mean tool calls", **{s: f"{res['systems'][s]['mean_tool_calls']:.2f}" for s in data}}]
    out = Path(a.out) if a.out else M / f"summary_{a.split}.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    for s in data:
        print("==", s)
        for k, v in res["systems"][s].items():
            print(f"  {k}: {pct(*v) if isinstance(v, tuple) and len(v) == 2 and v[1] else v}")
    print("by category:", json.dumps(res["by_category"], default=str))
    print("paired:", json.dumps({s: v.get("overall") for s, v in res["paired_vs_baseline"].items()}))


if __name__ == "__main__":
    main()
