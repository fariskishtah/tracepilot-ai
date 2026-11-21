"""Render markdown result tables from the saved JSON results and splice them into README/docs between markers
<!-- BEGIN:name --> ... <!-- END:name -->.  Usage: python scripts/render_results_md.py"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "artifacts" / "metrics"
J = lambda n: json.loads((M / n).read_text()) if (M / n).exists() else None
pct = lambda k, n: f"{k}/{n} ({100 * k / n:.1f}%)" if n else "n/a"


def tables() -> dict[str, str]:
    S, A, D = J("summary_test.json"), J("summary_ablation_test.json"), J("summary_dev.json")
    out = {}
    if S:
        st, fu = S["systems"]["standard"], S["systems"]["full"]
        ov = S["paired_vs_baseline"]["full"]["overall"]
        rows = ["| metric (test split, n=%d) | Standard RAG | TracePilot (agentic) |" % fu["n"], "|---|---|---|"]
        for r in S["table"]:
            rows.append(f"| {r['metric']} | {r['standard']} | {r['full']} |")
        for k in ["false_abstention (answerable)"]:
            rows.append(f"| {k} | {pct(*st[k])} | {pct(*fu[k])} |")
        rows.append(f"| gold-document citation hit | {pct(*st['citation_hit (answerable w/ gold docs)'])} | {pct(*fu['citation_hit (answerable w/ gold docs)'])} |")
        rows.append(f"| mean LLM calls / tokens | {st['mean_llm_calls']:.2f} / {st['mean_tokens']:.0f} | {fu['mean_llm_calls']:.2f} / {fu['mean_tokens']:.0f} |")
        rows.append("")
        rows.append(f"Paired difference in correctness: **{100 * ov['mean_diff']:+.1f} pp**, 95% paired-bootstrap CI **[{100 * ov['ci95'][0]:+.1f}, {100 * ov['ci95'][1]:+.1f}] pp** "
                    f"(TracePilot-only correct: {ov['b_only']}, Standard-only correct: {ov['a_only']}; "
                    f"{'interval excludes 0' if ov['significant'] else 'interval includes 0 — NOT statistically confirmed'}).")
        out["main"] = "\n".join(rows)
        cats = sorted(S["by_category"]["full"])
        rows = ["| category | n | Standard RAG | TracePilot | paired diff (pp) |", "|---|---|---|---|---|"]
        for c in cats:
            a, b = S["by_category"]["standard"][c], S["by_category"]["full"][c]
            d = S["paired_vs_baseline"]["full"].get(c, {})
            rows.append(f"| {c} | {b[1]} | {a[0]}/{a[1]} | {b[0]}/{b[1]} | {100 * d.get('mean_diff', 0):+.0f} |")
        out["categories"] = "\n".join(rows)
        rows = ["| planning / execution metric (test) | value |", "|---|---|",
                f"| correct execution path | {pct(*fu['tool_correct_path'])} |", f"| required-capability recall | {pct(*fu['tool_required_ok(recall)'])} |",
                f"| mean tool precision | {fu['mean_tool_precision']:.3f} |", f"| unnecessary-call rate | {pct(*fu['unnecessary_call_rate'])} |",
                f"| forbidden tool used | {pct(*fu['forbidden_tool_used'])} |", f"| plan completion | {pct(*fu['plan_completion'])} |",
                f"| requests that replanned | {pct(*fu['replanned_requests'])} |", f"| planner rule-fallback | {pct(*fu['planner_rules_fallback'])} |",
                f"| multi-step full completion (answer-level) | Standard {pct(*st['multi_step_full_completion'])} → TracePilot {pct(*fu['multi_step_full_completion'])} |",
                f"| latency mean / median / p95 (s) | Standard {st['latency']['mean']:.1f} / {st['latency']['median']:.1f} / {st['latency']['p95']:.1f} → TracePilot {fu['latency']['mean']:.1f} / {fu['latency']['median']:.1f} / {fu['latency']['p95']:.1f} |"]
        out["planning"] = "\n".join(rows)
    if A:
        rows = ["| system | n | correct | numeric | multi-step | mean latency (s) | tool calls |", "|---|---|---|---|---|---|---|"]
        for v in A["rows"].values():
            rows.append(f"| {v['label']} | {v['n']} | {pct(*v['correct'])} | {pct(*v['numeric'])} | {pct(*v['multi'])} | {v['lat']:.1f} | {v['calls']:.2f} |")
        rows += ["", A["note"]]
        out["ablation"] = "\n".join(rows)
    if D:
        st, fu = D["systems"]["standard"], D["systems"]["full"]
        out["dev"] = (f"Development split (n={fu['n']}, used for tuning — optimistic): Standard RAG {pct(*st['answer_correctness (all)'])}, TracePilot {pct(*fu['answer_correctness (all)'])}.")
    return out


def splice(path: Path, t: dict[str, str]) -> None:
    s = path.read_text()
    for k, v in t.items():
        s = re.sub(rf"(<!-- BEGIN:{k} -->)(.*?)(<!-- END:{k} -->)", lambda m: f"{m.group(1)}\n{v}\n{m.group(3)}", s, flags=re.S)
    path.write_text(s)


if __name__ == "__main__":
    t = tables()
    for f in [ROOT / "README.md", ROOT / "docs" / "EVALUATION.md", ROOT / "docs" / "BASELINE_COMPARISON.md", ROOT / "docs" / "ABLATION_STUDY.md"]:
        if f.exists():
            splice(f, t)
    print({k: len(v) for k, v in t.items()})
