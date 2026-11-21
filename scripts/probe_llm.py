"""Probe: native Ollama tool-calling vs schema-constrained planning for FIRST-TOOL selection on the DEV split.
Purpose: justify the planner design with a measurement (docs/AGENT_DESIGN.md). Reads nothing from the test split.
Usage: python scripts/probe_llm.py"""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import cfg  # noqa: E402
from src.evaluation.agent_metrics import CAP  # noqa: E402
from src.evaluation.benchmarking import load_questions  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "search_knowledge_base", "description": "Search course documents for facts, definitions, explanations.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "analyze_dataset", "description": "Exact statistics on a CSV dataset (mean, median, sum, count, lookup).", "parameters": {"type": "object", "properties": {"dataset_id": {"type": "string"}, "operation": {"type": "string"}, "column": {"type": "string"}}, "required": ["dataset_id", "operation"]}}},
    {"type": "function", "function": {"name": "calculate_expression", "description": "Evaluate an arithmetic expression.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]


def main():
    c = cfg("models")["llm"]
    items = load_questions("dev")
    ok = n = valid = 0
    t0 = time.time()
    for it in items:
        if not it["requires_tools"]:
            continue
        r = requests.post(f"{c['host']}/api/chat", json={"model": c["model"], "stream": False, "messages": [
            {"role": "system", "content": "Pick the single best first tool for the user's question."}, {"role": "user", "content": it["question"]}],
            "tools": TOOLS, "options": {"temperature": 0, "seed": 7, "num_ctx": 4096}}, timeout=120).json()
        calls = (r.get("message") or {}).get("tool_calls") or []
        n += 1
        if calls:
            valid += 1
            first = CAP.get(calls[0]["function"]["name"])
            first_ok = first in it["requires_tools"] or (first == "search" and "compare" in it["requires_tools"])
            ok += bool(first_ok)
    out = {"model": c["model"], "n": n, "valid_native_tool_calls": valid, "first_tool_acceptable": ok, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out))
    (ROOT / "artifacts" / "metrics" / "probe_native_tool_calling_dev.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
