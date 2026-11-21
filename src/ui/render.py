"""Pure rendering helpers (HTML/markdown) for the Studio. All content comes from real AgentResult/trace data."""
from __future__ import annotations

import html
import json

ICON = {"request_received": "▶", "question_analysis": "🔎", "plan_created": "🧭", "tool_selected": "🔧", "tool_result": "✔",
        "step_failed": "✖", "step_skipped": "⤼", "validation": "⚖", "replan": "↻", "replan_skipped": "⛔", "synthesis_started": "✍",
        "citation_check": "🔗", "answer": "💬", "abstain": "🚫", "completed": "■", "error": "⚠", "budget_exhausted": "⛔",
        "dataset_resolved": "🗂", "retrieval": "🔍"}
E = html.escape


def _kv(d: dict, skip=("seq", "t_ms", "type")) -> str:
    rows = []
    for k, v in d.items():
        if k in skip or v in (None, "", [], {}):
            continue
        s = json.dumps(v, default=str) if not isinstance(v, str) else v
        rows.append(f"<div class='kv'><span>{E(k)}</span><code>{E(s[:420])}</code></div>")
    return "".join(rows)


def event_html(ev: dict) -> str:
    t = ev["type"]
    cls = "ok"
    if t in ("step_failed", "error", "budget_exhausted", "replan_skipped", "abstain"):
        cls = "bad"
    elif t in ("replan", "step_skipped"):
        cls = "warn"
    elif t == "tool_result" and not ev.get("ok", True):
        cls = "bad"
    title = t.replace("_", " ")
    if t == "tool_selected":
        title = f"tool selected · {ev.get('tool')} ({ev.get('step')}, attempt {ev.get('attempt')})"
    elif t == "tool_result":
        title = f"tool result · {ev.get('tool')} ({ev.get('step')}) · {'ok' if ev.get('ok') else 'FAILED'} · {ev.get('ms')} ms"
    elif t == "plan_created":
        title = f"plan · {ev.get('question_type')} / {ev.get('complexity')} · source: {ev.get('source')}"
    elif t == "replan":
        title = f"replanning round {ev.get('round')}"
    body = _kv(ev)
    if t == "plan_created":
        steps = "".join(f"<li><b>{E(s['id'])}</b> {E(s['tool'])} — {E(s['goal'])}"
                        + (f" <i>(needs {', '.join(s['depends_on'])})</i>" if s['depends_on'] else "") + "</li>" for s in ev.get("subtasks", []))
        body = f"<ol class='steps'>{steps}</ol>" + _kv({k: v for k, v in ev.items() if k not in ("subtasks",)})
    return (f"<div class='tl {cls}'><div class='dot'>{ICON.get(t, '•')}</div><div class='card'>"
            f"<div class='ttl'><span>{E(title)}</span><small>+{ev.get('t_ms', 0)} ms</small></div>{body}</div></div>")


def timeline_html(events: list[dict]) -> str:
    if not events:
        return "<div class='empty'>No execution yet. Ask a question to record a real trace.</div>"
    return "<div class='timeline'>" + "".join(event_html(e) for e in events) + "</div>"


def evidence_table(result: dict | None) -> list[list]:
    if not result:
        return []
    cited = {c["id"] for c in result["citations"]}
    rows = []
    for i, e in result["evidence"].items():
        kind = {"document": "original document", "dataset": "dataset computation", "calculation": "calculation", "summary": "generated summary"}[e["kind"]]
        rows.append([i, "✔" if i in cited else "", kind, e["title"], e.get("location") or "", e.get("score") if e.get("score") is not None else "",
                     (e["text"][:140] + "…") if len(e["text"]) > 140 else e["text"]])
    rows.sort(key=lambda r: (r[1] != "✔", r[0]))
    return rows


def metrics_md(result: dict | None) -> str:
    if not result:
        return ""
    m = result["metrics"]
    tools = " → ".join(result["tool_sequence"]) or "(none)"
    return (f"**Stop reason:** `{result['stop_reason']}` · **latency:** {m['latency_s']} s · **tool calls:** {m['tool_calls']} · "
            f"**LLM calls:** {m['llm_calls']} · **replans:** {m['replans']} · **tokens:** {m['prompt_tokens']}+{m['completion_tokens']}\n\n**Tool sequence:** {tools}")


STATUS_COLOR = {"done": "#22c55e", "weak": "#f59e0b", "failed": "#ef4444", "skipped": "#94a3b8", "pending": "#64748b"}


def workflow_svg(result: dict | None) -> str:
    """Executed workflow graph reconstructed from the real plan + trace (not a fictional diagram)."""
    if not result or not result.get("plan"):
        return "<div class='empty'>Run a question to see its actual execution graph.</div>"
    plan = result["plan"]
    status = {}
    attempts = {}
    for e in result["trace"]:
        if e["type"] == "tool_result":
            status[e["step"]] = "done" if e["ok"] and e.get("sufficient", True) else ("weak" if e["ok"] else "failed")
            attempts[e["step"]] = attempts.get(e["step"], 0) + 1
        elif e["type"] in ("step_failed",):
            status[e["step"]] = "failed"
        elif e["type"] == "step_skipped":
            status[e["step"]] = "skipped"
    subs = plan["subtasks"]
    W, H = 220, 54
    # columns: planner -> level by dependency depth
    depth = {}
    for s in subs:
        depth[s["id"]] = 1 + max([depth.get(d, 0) for d in s["depends_on"]] + [0])
    maxd = max(depth.values())
    cols: dict[int, list] = {}
    for s in subs:
        cols.setdefault(depth[s["id"]], []).append(s)
    nrows = max(len(v) for v in cols.values())
    width, height = 140 + (maxd + 2) * (W + 50), 40 + nrows * (H + 26) + 40
    out = [f"<svg viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' style='width:100%;font-family:Inter,sans-serif'>"]
    def box(x, y, w, label, sub, color):
        return (f"<rect x='{x}' y='{y}' width='{w}' height='{H}' rx='10' fill='#111a33' stroke='{color}' stroke-width='2'/>"
                f"<text x='{x + 12}' y='{y + 22}' fill='#fff' font-size='13' font-weight='600'>{E(label[:28])}</text>"
                f"<text x='{x + 12}' y='{y + 42}' fill='#9fb0d0' font-size='11'>{E(sub[:34])}</text>")
    pos = {"planner": (20, height // 2 - H // 2)}
    out.append(box(20, pos["planner"][1], 130, "Planner", plan.get("question_type", ""), "#7c5cff"))
    for d, items in cols.items():
        for k, s in enumerate(items):
            x, y = 190 + (d - 1) * (W + 50), 30 + k * (H + 26) + (nrows - len(items)) * (H + 26) // 2
            pos[s["id"]] = (x, y)
            c = STATUS_COLOR.get(status.get(s["id"], "pending"))
            lab = f"{s['id']} · {s['tool']}" + (f" ×{attempts[s['id']]}" if attempts.get(s["id"], 0) > 1 else "")
            out.append(box(x, y, W, lab, s["goal"], c))
    for s in subs:
        x, y = pos[s["id"]]
        srcs = s["depends_on"] or ["planner"]
        for d in srcs:
            sx, sy = pos[d]
            sw = 130 if d == "planner" else W
            out.append(f"<path d='M{sx + sw} {sy + H // 2} C{sx + sw + 30} {sy + H // 2},{x - 30} {y + H // 2},{x} {y + H // 2}' stroke='#4b6bff' stroke-width='1.6' fill='none'/>")
    fx = 190 + maxd * (W + 50)
    out.append(box(fx, height // 2 - H // 2, 150, "Synthesis + citations", result["stop_reason"], "#22c55e" if not result["abstained"] else "#f59e0b"))
    for s in subs:
        x, y = pos[s["id"]]
        out.append(f"<path d='M{x + W} {y + H // 2} C{x + W + 30} {y + H // 2},{fx - 30} {height // 2},{fx} {height // 2}' stroke='#2b3a63' stroke-width='1.2' fill='none'/>")
    out.append("</svg>")
    legend = "".join(f"<span class='lg'><i style='background:{c}'></i>{k}</span>" for k, c in STATUS_COLOR.items())
    return "<div class='wf'>" + "".join(out) + f"<div class='legend'>{legend}</div></div>"
