"""Figures from REAL result files (artifacts/metrics) + the architecture diagram. Usage: python scripts/make_figures.py"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
M, F = ROOT / "artifacts" / "metrics", ROOT / "artifacts" / "figures"
F.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, VIOLET, GREY = "#0b1020", "#3b82f6", "#7c5cff", "#94a3b8"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def rows(name):
    p = M / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
    def box(x, y, w, h, t, c=BLUE, tc="white", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=c, ec="none"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold", wrap=True)
    def arr(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
    box(0.2, 2.6, 1.5, 1.0, "User\nquestion", NAVY)
    box(2.1, 2.6, 1.7, 1.0, "Question\nanalyzer\n(rule features)", "#1e3a8a")
    box(4.1, 2.6, 1.7, 1.0, "LLM planner\n(JSON schema)\n+ plan guard", VIOLET)
    box(6.1, 2.6, 1.9, 1.0, "Execution controller\n(state machine,\nbudgets)", BLUE)
    for x1, x2 in [(1.7, 2.1), (3.8, 4.1), (5.8, 6.1)]:
        arr(x1, 3.1, x2, 3.1)
    tools = [("search_knowledge_base", 5.1), ("analyze_dataset", 4.45), ("calculate_expression", 3.8), ("compare, summarize, context", 1.45), ("refine_search_query", 0.8)]
    for t, y in tools:
        box(8.6, y, 2.2, 0.5, t, "#134e4a", fs=6.3 if len(t) > 22 else 7.5)
        arr(8.0, 3.1, 8.6, y + 0.25)
    box(6.1, 0.9, 1.9, 0.9, "Evidence store\nS# D# C# G# ids", "#334155")
    arr(8.6, 2.0, 8.0, 1.4); arr(7.05, 2.6, 7.05, 1.8)
    box(4.1, 0.9, 1.7, 0.9, "Validate\n(sufficiency)\n→ replan ≤2", "#92400e")
    arr(6.1, 1.35, 5.8, 1.35); arr(4.95, 1.8, 4.95, 2.6)
    box(2.1, 0.9, 1.7, 0.9, "Synthesis\n(untrusted\nevidence)", "#1e3a8a")
    arr(4.1, 1.35, 3.8, 1.35)
    box(0.2, 0.9, 1.5, 0.9, "Citation check\n→ answer /\nabstention", "#166534")
    arr(2.1, 1.35, 1.7, 1.35)
    ax.text(0.2, 5.6, "TracePilot AI — Agentic RAG architecture", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(0.2, 5.25, "planner → tools → evidence → validated, cited answer (all steps recorded in a structured trace)", fontsize=9, color=GREY)
    fig.savefig(F / "architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def categories(split):
    std, full = rows(f"{split}_standard"), rows(f"{split}_full")
    if not std or not full:
        return
    cats = sorted({r["category"] for r in full})
    def acc(rs, c):
        r = [x for x in rs if x["category"] == c]
        return 100 * sum(x["correct"] for x in r) / len(r) if r else 0
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x - w / 2, [acc(std, c) for c in cats], w, label="Standard RAG", color=GREY)
    ax.bar(x + w / 2, [acc(full, c) for c in cats], w, label="TracePilot (agentic)", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={sum(1 for r in full if r['category'] == c)})" for c in cats], fontsize=8)
    ax.set_ylabel("answer correctness (%)"); ax.set_ylim(0, 105); ax.legend(frameon=False)
    ax.set_title(f"Per-category correctness — {split} split (rubric-based)")
    fig.savefig(F / f"category_{split}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def latency(split):
    sysn = [("standard", "Standard RAG"), ("full", "Agentic")]
    data = [([r["latency_s"] for r in rows(f"{split}_{s}")], n) for s, n in sysn]
    data = [d for d in data if d[0]]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([d[0] for d in data], tick_labels=[d[1] for d in data], showfliers=True)
    ax.set_ylabel("latency per question (s)"); ax.set_title(f"Latency — {split} split")
    fig.savefig(F / f"latency_{split}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def tools(split):
    r = rows(f"{split}_full")
    if not r:
        return
    c = Counter(t for x in r for t in x["tool_sequence"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    k = sorted(c, key=c.get)
    ax.barh(k, [c[i] for i in k], color=VIOLET)
    ax.set_xlabel("calls"); ax.set_title(f"Tool usage frequency — {split} split ({len(r)} questions)")
    fig.savefig(F / f"tool_usage_{split}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def ablation(split):
    names = [("standard", "A Standard RAG"), ("llm_no_numeric_tools", "B agent, retrieval tools only"), ("rules_all_tools", "C rules planner, all tools"),
             ("llm_no_replan", "D LLM planner, no replan"), ("full", "F full (LLM + replan)")]
    vals = []
    for s, n in names:
        r = rows(f"{split}_{s}")
        if r:
            vals.append((n, 100 * sum(x["correct"] for x in r) / len(r), len(r)))
    if len(vals) < 2:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.barh([v[0] for v in vals][::-1], [v[1] for v in vals][::-1], color=[BLUE if "full" in v[0].lower() else GREY for v in vals][::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v[1] + 1, i, f"{v[1]:.1f}% (n={v[2]})", va="center", fontsize=8)
    ax.set_xlim(0, 115); ax.set_xlabel("answer correctness (%)"); ax.set_title(f"Ablation — {split} split")
    fig.savefig(F / f"ablation_{split}.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    architecture()
    for s in ("dev", "test"):
        categories(s); latency(s); tools(s); ablation(s)
    print(sorted(p.name for p in F.glob("*.png")))
