"""TracePilot Studio (Gradio). Every displayed event comes from real backend execution."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from src.config import ROOT, cfg, documents_dir
from src.ingestion.loaders import IngestionError
from src.service import Service, load_metrics
from src.tools.base import EvidenceStore, ToolContext
from src.ui import render as R
from src.ui.theme import CSS

EXAMPLES = [
    "What is supervised learning?",
    "Compare supervised and unsupervised learning.",
    "Calculate the sample standard deviation of the price_usd column in the product prices dataset.",
    "Compare the average and median price in the product prices dataset, then explain why they differ.",
    "Explain overfitting, compare L2 regularization and early stopping, then summarize their advantages.",
    "Find the definition of overfitting, retrieve two mitigation techniques, compare them, and calculate the difference between the validation accuracies of run R02 and run R04 in the regularization experiments CSV.",
    "What is the capital of France?",
]


def build_app(svc: Service | None = None) -> gr.Blocks:
    svc = svc or Service()
    courses = svc.kb.courses() or ["ml"]
    tool_names = svc.agent().registry.all_names()

    with gr.Blocks(title="TracePilot AI", css=CSS, theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate").set(
            body_background_fill="#0b1020", block_background_fill="#111a33", block_border_color="#24305a",
            input_background_fill="#0b1230", body_text_color="#eaf0ff", block_label_text_color="#9fb0d0",
            checkbox_label_background_fill="#1b2652", checkbox_label_background_fill_hover="#24305a", checkbox_label_background_fill_selected="#2a3a7a",
            checkbox_label_text_color="#eaf0ff", checkbox_label_text_color_selected="#ffffff", checkbox_label_border_color="#3b4a85",
            table_even_background_fill="#111a33", table_odd_background_fill="#0d1530", table_border_color="#24305a", table_text_color="#eaf0ff",
            table_row_focus="#1b2652", code_background_fill="#0b1230")) as app:
        gr.HTML("<div class='tp-hero'><h1>TracePilot AI</h1><p>Agentic RAG &amp; Intelligent Knowledge Orchestration</p>"
                f"<span class='pill'>local LLM · {svc.llm.model}</span><span class='pill'>MiniLM embeddings · ChromaDB</span>"
                "<span class='pill'>planner → tools → evidence → cited answer</span></div>")
        last = gr.State(None)

        # ------------------------------------------------------------ Agentic Q&A
        with gr.Tab("Agentic Q&A"):
            with gr.Row():
                with gr.Column(scale=3):
                    q = gr.Textbox(label="Your question", lines=3, placeholder="Ask a multi-step question about the course documents or datasets…")
                    with gr.Row():
                        scope = gr.CheckboxGroup(courses, value=courses, label="Knowledge base (courses)", elem_classes=["tp-col"])
                        tools = gr.CheckboxGroup(tool_names, value=svc.agent().registry.names(), label="Enabled tools", elem_classes=["tp-col"])
                    ex = gr.Dataset(components=[q], samples=[[e] for e in EXAMPLES], label="Examples", type="values")
                    go = gr.Button("Run agent", variant="primary")
                with gr.Column(scale=4):
                    status = gr.Markdown("*Idle.*")
                    answer = gr.Markdown(elem_classes="answer-box", value="*The answer will appear here, with citations.*")
                    meta = gr.Markdown()
            with gr.Accordion("Live execution trace", open=True):
                live = gr.HTML(R.timeline_html([]))
            ev_tbl = gr.Dataframe(headers=["id", "cited", "type", "source", "location", "score", "excerpt"], label="Evidence used", wrap=True, interactive=False)

            def run(question, scope_v, tools_v, progress=gr.Progress()):
                if not question or not question.strip():
                    yield "*Please enter a question.*", "", "", R.timeline_html([]), [], None
                    return
                if not scope_v:
                    yield "*Select at least one course.*", "", "", R.timeline_html([]), [], None
                    return
                events, res = [], None
                yield "**Running…** (local LLM; the first request loads the model)", "", "", R.timeline_html([]), [], None
                with svc.lock:
                    for e in svc.agent().run_iter(question, scope_v, tools_v):
                        if e["type"] == "final":
                            res = e["result"].to_dict()
                        else:
                            events.append(e)
                            yield f"**Step {len(events)}:** {e['type'].replace('_', ' ')}", "", "", R.timeline_html(events), [], None
                yield (f"**Done** in {res['metrics']['latency_s']} s", res["answer"], R.metrics_md(res), R.timeline_html(res["trace"]),
                       R.evidence_table(res), res)
            go.click(run, [q, scope, tools], [status, answer, meta, live, ev_tbl, last])
            q.submit(run, [q, scope, tools], [status, answer, meta, live, ev_tbl, last])
            ex.click(lambda v: v[0], ex, q)

        # ------------------------------------------------------------ Execution trace
        with gr.Tab("Execution Trace"):
            gr.Markdown("Structured, action-level execution log of the **last request** (plan, tool calls, results, replanning, citation check). "
                        "It contains no hidden model reasoning.")
            btn = gr.Button("Show last request")
            tl = gr.HTML(R.timeline_html([]))
            dl = gr.File(label="Download trace JSON")
            raw = gr.JSON(label="Raw trace", visible=True)

            def show_trace(res):
                if not res:
                    return R.timeline_html([]), None, None
                f = Path(tempfile.mkdtemp()) / f"trace_{res['request_id']}.json"
                f.write_text(json.dumps(res, indent=1, default=str))
                return R.timeline_html(res["trace"]), str(f), res["trace"]
            btn.click(show_trace, last, [tl, dl, raw])

        # ------------------------------------------------------------ Tool explorer
        with gr.Tab("Tool Explorer"):
            specs = {s["name"]: s for s in svc.agent().registry.specs()}
            gr.Dataframe(value=[[s["name"], s["description"], s["output"], "read-only" if s["read_only"] else "write"] for s in specs.values()],
                         headers=["tool", "purpose", "output", "access"], wrap=True, interactive=False)
            sel = gr.Dropdown(list(specs), value=list(specs)[0], label="Tool")
            schema = gr.JSON(label="Input schema (Pydantic)")
            args = gr.Code(label="Arguments (JSON) - validated against the schema before execution", language="json",
                           value=json.dumps(list(specs.values())[0]["examples"][0], indent=2))
            run_t = gr.Button("Execute safely", variant="primary")
            out = gr.JSON(label="Result")
            note = gr.Markdown("Manual execution uses a fresh evidence store. Tools that read earlier evidence (e.g. `compare_evidence`) need S-ids from a search run first - "
                               "run `search_knowledge_base` first in the same session to see how S-ids are minted.")
            manual_ctx = {"ctx": None}

            def pick(n):
                s = specs[n]
                return s["input_schema"], json.dumps(s["examples"][0] if s["examples"] else {}, indent=2)

            def exec_tool(n, a):
                try:
                    raw_args = json.loads(a)
                except json.JSONDecodeError as e:
                    return {"ok": False, "error": f"invalid JSON: {e}"}
                if manual_ctx["ctx"] is None:
                    manual_ctx["ctx"] = ToolContext(svc.kb, svc.retriever, svc.llm, EvidenceStore())
                with svc.lock:
                    r = svc.agent().registry.execute(n, raw_args, manual_ctx["ctx"])
                return {"ok": r.ok, "error": r.error, "error_type": r.error_type, "summary": r.summary, "evidence_ids": r.evidence_ids,
                        "duration_ms": r.duration_ms, "data": r.data}
            sel.change(pick, sel, [schema, args])
            app.load(lambda: specs[list(specs)[0]]["input_schema"], None, schema)
            run_t.click(exec_tool, [sel, args], out)

        # ------------------------------------------------------------ Knowledge library
        with gr.Tab("Knowledge Library"):
            def docs_df():
                d = svc.kb.list_documents()
                return pd.DataFrame([{"doc_id": x["doc_id"], "course": x["course_id"], "type": x["file_type"], "chunks": x["n_chunks"],
                                      "pages/rows": x["n_pages"] or x["n_rows"] or "", "status": x["status"], "origin": x["origin"],
                                      "sha256": x["sha256"][:10]} for x in d])
            tbl = gr.Dataframe(docs_df(), interactive=False, label="Indexed documents")
            with gr.Row():
                up = gr.File(label="Upload (.pdf .docx .csv .txt, max 5 MB)", file_types=[".pdf", ".docx", ".csv", ".txt"])
                up_course = gr.Textbox(label="Course id", value="uploads")
                up_btn = gr.Button("Upload & index", variant="primary")
            msg = gr.Markdown()
            with gr.Row():
                doc_sel = gr.Dropdown([x["doc_id"] for x in svc.kb.list_documents()], label="Document")
                re_btn, del_btn, insp_btn = gr.Button("Reindex"), gr.Button("Delete"), gr.Button("Inspect chunks")
            insp = gr.Dataframe(label="Chunks", wrap=True, interactive=False)

            def refresh():
                return docs_df(), gr.update(choices=[x["doc_id"] for x in svc.kb.list_documents()])

            def upload(f, course):
                if f is None:
                    return "Choose a file first.", *refresh()
                try:
                    with svc.lock:
                        r = svc.kb.ingest_file(Path(f), course or "uploads", origin="upload", store_copy=True)
                    m = f"**{r.status}**: `{r.doc_id}` ({r.n_chunks} chunks)"
                except IngestionError as e:
                    m = f"❌ Rejected: {e}"
                return m, *refresh()

            def delete(d):
                ok = svc.kb.delete_document(d) if d else False
                return ("Deleted." if ok else "Nothing deleted."), *refresh()

            def reindex(d):
                try:
                    with svc.lock:
                        r = svc.kb.reindex_document(d)
                    return f"Reindexed `{r.doc_id}` ({r.n_chunks} chunks).", *refresh()
                except Exception as e:
                    return f"❌ {type(e).__name__}: {e}", *refresh()

            def inspect(d):
                rows = svc.kb.store.get(where={"doc_id": d}) if d else []
                rows.sort(key=lambda r: r["meta"].get("chunk_index", 0))
                return pd.DataFrame([{"chunk_id": r["chunk_id"], "page": r["meta"].get("page"), "section": r["meta"].get("section"),
                                      "rows": f"{r['meta'].get('row_start')}-{r['meta'].get('row_end')}" if r["meta"].get("row_start") else "",
                                      "text": r["text"][:300]} for r in rows])
            up_btn.click(upload, [up, up_course], [msg, tbl, doc_sel])
            del_btn.click(delete, doc_sel, [msg, tbl, doc_sel])
            re_btn.click(reindex, doc_sel, [msg, tbl, doc_sel])
            insp_btn.click(inspect, doc_sel, insp)

        # ------------------------------------------------------------ Evidence inspector
        with gr.Tab("Evidence Inspector"):
            gr.Markdown("Inspect any item the last request retrieved or computed. **Generated summaries are labelled and never count as original evidence.**")
            load_ev = gr.Button("Load evidence of last request")
            ev_sel = gr.Dropdown([], label="Evidence id")
            ev_md = gr.Markdown()
            ev_state = gr.State({})

            def load_evidence(res):
                if not res:
                    return gr.update(choices=[]), {}, "No request yet."
                ids = list(res["evidence"])
                return gr.update(choices=ids, value=ids[0] if ids else None), res, f"{len(ids)} evidence items."

            def show_ev(i, res):
                if not i or not res or i not in res["evidence"]:
                    return ""
                e = res["evidence"][i]
                cited = i in {c["id"] for c in res["citations"]}
                kind = {"document": "📄 Original document passage", "dataset": "🧮 Deterministic dataset computation",
                        "calculation": "➗ Deterministic calculation", "summary": "⚠ Generated summary (derived, not original evidence)"}[e["kind"]]
                d = e.get("data") or {}
                prov = {k: v for k, v in d.items() if k in ("file", "file_sha256", "operation", "column", "rows_used", "rows_total", "filters", "expression", "inputs", "input_evidence", "page", "section", "filename")}
                return (f"### [{i}] {e['title']}\n{kind} · cited in answer: **{'yes' if cited else 'no'}**\n\n"
                        f"- document: `{e.get('doc_id')}` · course: `{e.get('course_id')}` · location: **{e.get('location')}** · cosine: {e.get('score')}\n"
                        f"- derived from: {e.get('derived_from') or '-'}\n- provenance: `{json.dumps(prov, default=str)}`\n\n> {e['text']}")
            load_ev.click(load_evidence, last, [ev_sel, ev_state, ev_md])
            ev_sel.change(show_ev, [ev_sel, ev_state], ev_md)

        # ------------------------------------------------------------ Workflow
        with gr.Tab("Agent Workflow"):
            gr.Markdown("Execution graph reconstructed from the **actual plan and trace** of the last request. Border colour = step status; `×n` = attempts after replanning.")
            wf_btn = gr.Button("Render last execution")
            wf = gr.HTML(R.workflow_svg(None))
            gr.Markdown("**Controller state machine:** `ANALYZE → PLAN → EXECUTE → VALIDATE → (REPLAN → EXECUTE)* → SYNTHESIZE → CITATION_CHECK → DONE` "
                        f"· budgets: ≤{cfg('agent')['budgets']['max_tool_calls']} tool calls, ≤{cfg('agent')['budgets']['max_replans']} replans, "
                        f"≤{cfg('agent')['budgets']['max_attempts_per_subtask']} attempts/subtask.")
            wf_btn.click(R.workflow_svg, last, wf)

        # ------------------------------------------------------------ Standard vs Agentic
        with gr.Tab("Standard vs Agentic RAG"):
            cq = gr.Textbox(label="Question (same question, same model, same retriever)", lines=2)
            cscope = gr.CheckboxGroup(courses, value=courses, label="Courses")
            cb = gr.Button("Run both", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Standard RAG (retrieve once → answer)")
                    s_ans, s_meta = gr.Markdown(), gr.Markdown()
                with gr.Column():
                    gr.Markdown("### TracePilot Agentic RAG")
                    a_ans, a_meta = gr.Markdown(), gr.Markdown()
            cmp_tbl = gr.Dataframe(label="Comparison", interactive=False)

            def compare(question, sc):
                if not question or not sc:
                    return "", "", "", "", pd.DataFrame()
                with svc.lock:
                    s = svc.baseline.run(question, sc).to_dict()
                    a = svc.agent().run(question, sc).to_dict()
                def row(name, f):
                    return {"metric": name, "Standard RAG": f(s), "Agentic RAG": f(a)}
                df = pd.DataFrame([
                    row("latency (s)", lambda r: r["metrics"]["latency_s"]), row("LLM calls", lambda r: r["metrics"]["llm_calls"]),
                    row("tool calls", lambda r: r["metrics"]["tool_calls"]), row("tokens (prompt+completion)", lambda r: r["metrics"]["prompt_tokens"] + r["metrics"]["completion_tokens"]),
                    row("sources retrieved", lambda r: sum(1 for e in r["evidence"].values() if e["kind"] == "document")),
                    row("citations in answer", lambda r: len(r["citations"])), row("abstained", lambda r: r["abstained"]),
                    row("citation support rate", lambda r: r["validation"].get("support_rate", "-")), row("tool sequence", lambda r: " → ".join(r["tool_sequence"]))])
                return s["answer"], R.metrics_md(s), a["answer"], R.metrics_md(a), df
            cb.click(compare, [cq, cscope], [s_ans, s_meta, a_ans, a_meta, cmp_tbl])

        # ------------------------------------------------------------ Analytics
        with gr.Tab("Analytics"):
            gr.Markdown("Computed from **recorded traces** of requests served by this installation and, when available, from the saved benchmark results in `artifacts/metrics/`.")
            ab = gr.Button("Refresh analytics")
            an_md = gr.Markdown()
            an_tools = gr.Dataframe(label="Tool usage (recorded requests)", interactive=False)
            bench = gr.Dataframe(label="Benchmark summary (artifacts/metrics/summary.json)", interactive=False)

            def analytics():
                tr = [t for t in svc.recent_traces(500) if t.get("system") == "agentic"]
                if not tr:
                    md = "No agentic requests recorded yet."
                    cnt = pd.DataFrame()
                else:
                    import numpy as np
                    lat = np.array([t["metrics"]["latency_s"] for t in tr])
                    from collections import Counter
                    cnt = Counter(x for t in tr for x in t["tool_sequence"])
                    ok = sum(1 for t in tr if t["stop_reason"] == "answered")
                    md = (f"**Requests:** {len(tr)} · **answered:** {ok} · **abstained:** {sum(t['abstained'] for t in tr)} · "
                          f"**avg tool calls:** {np.mean([t['metrics']['tool_calls'] for t in tr]):.2f} · **replanned:** {sum(1 for t in tr if t['metrics']['replans'])} · "
                          f"**tool failures:** {sum(1 for t in tr for e in t['trace'] if e['type']=='tool_result' and not e['ok'])} · "
                          f"**mean latency:** {lat.mean():.1f} s · **p95:** {np.percentile(lat, 95):.1f} s")
                    cnt = pd.DataFrame(sorted(cnt.items(), key=lambda x: -x[1]), columns=["tool", "calls"])
                s = load_metrics("summary.json")
                b = pd.DataFrame(s["table"]) if s and "table" in s else pd.DataFrame([{"info": "Run scripts/evaluate.py to populate"}])
                return md, cnt, b
            ab.click(analytics, None, [an_md, an_tools, bench])

        # ------------------------------------------------------------ Failure analysis
        with gr.Tab("Failure Analysis"):
            fa = ROOT / "docs" / "FAILURE_ANALYSIS.md"
            gr.Markdown(fa.read_text() if fa.exists() else "Failure analysis is generated from the evaluation traces (`scripts/evaluate.py`, `scripts/failure_analysis.py`).")
    return app
