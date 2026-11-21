"""Ablation table (test split) from saved result files + paired bootstrap vs the full system. Usage: python scripts/summarize_ablation.py [--split test]"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.benchmarking import frac, latency_stats, mean, paired_bootstrap  # noqa: E402

M = ROOT / "artifacts" / "metrics"
SYSTEMS = [("standard", "A  Standard RAG"), ("llm_no_numeric_tools", "B  LLM planner, retrieval tools only (no calc/data), no replan"),
           ("rules_all_tools", "C  rule planner, all tools, no replan"), ("llm_no_replan", "D/E  LLM planner, all tools, no replan"), ("full", "F  full: LLM planner + bounded replanning")]


def load(split, s):
    p = M / f"{split}_{s}.jsonl"
    rows = {}
    if p.exists():
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if r["multi_step"] and r["answerable"]:
                r["full_completion"] = bool(r["correct"] and (r["cit_citation_hit"] if r.get("cit_citation_hit") is not None else True))
            rows[r["id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    data = {s: load(a.split, s) for s, _ in SYSTEMS}
    ids = sorted(set.intersection(*[set(d) for d in data.values() if d]))
    out = {"split": a.split, "n_common": len(ids), "rows": {}, "paired_vs_full": {}}
    for s, label in SYSTEMS:
        if not data[s]:
            continue
        rs = [data[s][i] for i in ids]
        ms = [r for r in rs if r["multi_step"] and r["answerable"]]
        num = [r for r in rs if r.get("numeric_ok") is not None]
        out["rows"][s] = {"label": label, "n": len(rs), "correct": frac(rs, "correct"), "numeric": frac(num, "numeric_ok"), "multi": frac(ms, "full_completion"),
                          "lat": latency_stats(rs)["mean"], "calls": mean([r["tool_calls"] for r in rs]), "llm_calls": mean([r["llm_calls"] for r in rs])}
    if data["full"]:
        for s, _ in SYSTEMS:
            if s != "full" and data[s]:
                out["paired_vs_full"][s] = paired_bootstrap([data[s][i]["correct"] * 1.0 for i in ids], [data["full"][i]["correct"] * 1.0 for i in ids], n=3000)
    pv = out["paired_vs_full"]
    out["note"] = ("Paired bootstrap of (full − system) in correctness: " + "; ".join(
        f"vs {s}: {100 * v['mean_diff']:+.1f} pp, 95% CI [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]" for s, v in pv.items()) +
        ". Experiments D and E coincide (an LLM planner without replanning); intervals that include zero mean the component's contribution is not confirmed on this benchmark.")
    (M / f"summary_ablation_{a.split}.json").write_text(json.dumps(out, indent=2, default=str))
    for s, r in out["rows"].items():
        print(f"{r['label']:<62} n={r['n']} correct={r['correct']} numeric={r['numeric']} multi={r['multi']} lat={r['lat']:.1f}s calls={r['calls']:.2f}")
    print(out["note"])


if __name__ == "__main__":
    main()
