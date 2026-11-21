"""Run a system on an evaluation split. Resumable. Usage:
   python scripts/evaluate.py --system agentic|standard|<ablation> --split dev|test [--limit N]
Systems: standard | full | rules_all_tools | llm_no_numeric_tools | llm_no_replan   (see docs/ABLATION_STUDY.md)"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agents.controller import AgentConfig, TracePilotAgent  # noqa: E402
from src.config import cfg  # noqa: E402
from src.evaluation.benchmarking import load_questions, score_item  # noqa: E402
from src.service import Service  # noqa: E402

RET_ONLY = ["search_knowledge_base", "retrieve_source_context", "summarize_evidence", "compare_evidence", "refine_search_query"]
SYSTEMS = {
    "full": dict(planner_mode="llm", tools=None, replan=True),
    "llm_no_replan": dict(planner_mode="llm", tools=None, replan=False),
    "rules_all_tools": dict(planner_mode="rules", tools=None, replan=False),
    "llm_no_numeric_tools": dict(planner_mode="llm", tools=RET_ONLY, replan=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="full")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    svc = Service()
    items = load_questions(a.split)
    if a.ids:
        want = set(a.ids.split(","))
        items = [i for i in items if i["id"] in want]
    if a.limit:
        items = items[: a.limit]
    name = f"{a.split}_{a.system}{a.tag}"
    out = ROOT / "artifacts" / "metrics" / f"{name}.jsonl"
    tdir = ROOT / "artifacts" / "execution_traces" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    agent = None if a.system == "standard" else TracePilotAgent(svc.kb, svc.retriever, svc.llm, AgentConfig(name=a.system, **SYSTEMS[a.system]), trace_dir=tdir)
    svc.llm.chat("You are helpful.", "Say ok.", max_tokens=3)  # warm the model so first-question latency is not inflated
    for k, it in enumerate(items):
        if it["id"] in done:
            continue
        scope = it["course_scope"]
        t = time.time()
        res = (svc.baseline.run(it["question"], scope) if agent is None else agent.run(it["question"], scope)).to_dict()
        row = score_item(it, res, agentic=agent is not None)
        row["answer"] = res["answer"]
        row["tool_sequence"] = res["tool_sequence"]
        row["retrieved_docs"] = sorted({e["doc_id"] for e in res["evidence"].values() if e.get("doc_id")})
        with out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if agent is None:
            (tdir / f"{res['request_id']}.json").write_text(json.dumps(res, indent=1, default=str))
        print(f"[{k + 1}/{len(items)}] {it['id']} {'OK ' if row['correct'] else 'BAD'} {row['latency_s']:>5}s tools={res['tool_sequence']}", flush=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "uncommitted"
    except Exception:
        commit = "n/a"
    meta = {"system": a.system, "split": a.split, "llm": svc.llm.model, "embedding": cfg("models")["embeddings"]["model"], "temperature": 0, "seed": cfg("models")["llm"]["seed"],
            "registry_version": cfg("agent")["registry_version"], "agent_config": cfg("agent"), "retrieval_config": cfg("retrieval"), "date": time.strftime("%Y-%m-%d %H:%M"),
            "dataset": json.loads((ROOT / "data/evaluation/metadata.json").read_text()), "code_commit": commit, "n": len(items)}
    (ROOT / "artifacts" / "metrics" / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
