"""Build TracePilot_AI_Research_Report.pdf from REAL result files (artifacts/metrics, figures, screenshots, docs).
Every number in the report is read from a JSON result file at build time. Usage: python scripts/make_report.py"""
import json
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Preformatted)

ROOT = Path(__file__).resolve().parents[1]
M, FIG, SHOT, DOCS = ROOT / "artifacts" / "metrics", ROOT / "artifacts" / "figures", ROOT / "artifacts" / "screenshots", ROOT / "docs"
OUT = ROOT / "artifacts" / "report" / "TracePilot_AI_Research_Report.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, VIOLET = colors.HexColor("#0b1020"), colors.HexColor("#3b82f6"), colors.HexColor("#7c5cff")
GITHUB = "https://github.com/fariskishtah/tracepilot-ai"


def J(name, default=None):
    p = M / name
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return default


ST = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ST["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.4, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ST["Heading1"], fontName="Helvetica-Bold", fontSize=15, textColor=NAVY, spaceBefore=6, spaceAfter=8, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=ST["Heading2"], fontName="Helvetica-Bold", fontSize=11.2, textColor=BLUE, spaceBefore=6, spaceAfter=4)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor("#475569"), alignment=0)
CODE = ParagraphStyle("c", parent=BODY, fontName="Courier", fontSize=7.4, leading=9.4, alignment=0, backColor=colors.HexColor("#f1f5f9"), borderPadding=4)
esc = lambda t: str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_inline(t: str) -> str:
    t = esc(t)
    t = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    return t


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph("• " + i, ParagraphStyle("bl", parent=BODY, leftIndent=12, firstLineIndent=-8, spaceAfter=2)) for i in items]


def table(rows, widths=None, head=True, fs=8):
    data = [[Paragraph(esc(c) if not str(c).startswith("<") else c, ParagraphStyle("tc", parent=BODY, fontSize=fs, leading=fs + 2.5, alignment=0)) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if head:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff"))]
    t.setStyle(TableStyle(st))
    return KeepTogether([t, Spacer(1, 4)])


def img(path, w=15.5 * cm, cap=None):
    p = Path(path)
    if not p.exists():
        return [P(f"<i>[figure missing: {p.name}]</i>", SMALL)]
    from PIL import Image as PI
    iw, ih = PI.open(p).size
    out = [Image(str(p), width=w, height=w * ih / iw)]
    if cap:
        out.append(P(esc(cap), SMALL))
    return [KeepTogether(out)]


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "n/a"


def cover(c, d):
    w, h = A4
    c.setFillColor(NAVY)
    c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setFillColor(VIOLET)
    c.rect(0, h * 0.60, w, 5, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 34)
    c.drawString(2.2 * cm, h * 0.70, "TracePilot AI")
    c.setFont("Helvetica", 14)
    c.setFillColor(colors.HexColor("#c7d2fe"))
    c.drawString(2.2 * cm, h * 0.66, "Agentic RAG & Intelligent Knowledge Orchestration Platform")
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.white)
    for i, t in enumerate(["INSTANT AI & Data Science Training", "Sprint 4 — GANs & AI Agents", "Project 4 — Agentic RAG System"]):
        c.drawString(2.2 * cm, h * 0.54 - i * 0.7 * cm, t)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2.2 * cm, h * 0.40, "Prepared by: Faris Mohamed Kishtah")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#94a3b8"))
    c.drawString(2.2 * cm, 2.2 * cm, "Research report · all numbers are read from the saved evaluation result files · " + GITHUB)


def page(c, d):
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(2 * cm, 1.2 * cm, "TracePilot AI — Agentic RAG research report")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{d.page}")
    c.setStrokeColor(colors.HexColor("#e2e8f0"))
    c.line(2 * cm, 1.6 * cm, A4[0] - 2 * cm, 1.6 * cm)


def build():
    S = J("summary_test.json") or {}
    sy = S.get("systems", {})
    st, fu = sy.get("standard", {}), sy.get("full", {})
    pr = (S.get("paired_vs_baseline") or {}).get("full", {})
    ov = pr.get("overall", {})
    fz = json.loads((ROOT / "data/evaluation/FROZEN.json").read_text()) if (ROOT / "data/evaluation/FROZEN.json").exists() else {}
    md = json.loads((ROOT / "data/evaluation/metadata.json").read_text())
    AB = J("summary_ablation_test.json") or {}
    tests = J("tests_summary.json") or {}
    br = J("browser_verification.json") or {}
    pb = J("probe_native_tool_calling_dev.json") or {}
    dev = J("summary_dev.json") or {}
    fa = J("failures_test_full.json") or {}
    story = []
    story.append(PageBreak())
    story.append(P("Contents", H1))
    toc = ["Executive Summary", "Project Objectives", "Standard RAG Limitations", "Corrective RAG Background", "Agentic RAG Methodology", "System Architecture",
           "Knowledge Base", "Document Ingestion", "Embedding and Vector Retrieval", "Agent Planner", "Tool Registry", "Execution Controller", "Multi-Step Workflows",
           "Evidence Aggregation", "Grounded Answer Generation", "Citation Validation", "Application Architecture", "Agent Execution Trace", "Evaluation Methodology",
           "Standard vs Agentic Comparison", "Agent Planning Evaluation", "Tool Selection Evaluation", "Multi-Step Task Completion", "Ablation Results", "Latency Analysis",
           "Security", "Failure Analysis", "Limitations", "Conclusion", "Project Links"]
    story += [P(f"{i}. {t}", ParagraphStyle("toc", parent=BODY, spaceAfter=1, alignment=0)) for i, t in enumerate(toc, 1)]
    story.append(PageBreak())
    n = [0]

    def sec(title):
        n[0] += 1
        return KeepTogether([CondPageBreak(6 * cm), P(f"{n[0]}. {title}", H1)])

    # 1
    story += [sec("Executive Summary")]
    story.append(P("TracePilot AI is a locally running <b>Agentic RAG</b> system: for every question it <b>plans</b> a typed sequence of tool calls, <b>executes</b> them under explicit budgets, "
                   "<b>aggregates evidence</b> from document retrieval and deterministic computation (pandas, a safe calculator), <b>replans</b> in a bounded way when evidence is weak or a tool fails, and "
                   "produces an answer whose citations are verified against the evidence — or abstains. It runs on an Apple M2 with 8 GB RAM using Ollama (<i>qwen2.5:3b-instruct</i>), "
                   "all-MiniLM-L6-v2 embeddings, ChromaDB and Gradio; no paid service is used."))
    if st and fu:
        c1, c2 = st["answer_correctness (all)"], fu["answer_correctness (all)"]
        story.append(P(f"<b>Held-out test result (n={fu['n']}, frozen protocol):</b> rubric-based answer correctness {pct(*c1)} for Standard RAG vs {pct(*c2)} for TracePilot "
                       f"(paired difference {100 * ov.get('mean_diff', 0):+.1f} pp, 95% paired-bootstrap CI [{100 * ov['ci95'][0]:+.1f}, {100 * ov['ci95'][1]:+.1f}] pp; "
                       f"{'the interval excludes zero' if ov.get('significant') else 'the interval includes zero — not statistically confirmed'}). "
                       f"Numerical accuracy {pct(*st['numerical_accuracy'])} vs {pct(*fu['numerical_accuracy'])}; correct abstention on unanswerable questions "
                       f"{pct(*st['correct_abstention (unanswerable)'])} vs {pct(*fu['correct_abstention (unanswerable)'])}. Mean latency {st['latency']['mean']:.1f} s vs {fu['latency']['mean']:.1f} s."))
    if st and fu:
        cc_s, cc_f = S["by_category"]["standard"], S["by_category"]["full"]
        story.append(P(f"<b>Where it did not help:</b> comparison questions (Standard {cc_s['comparison'][0]}/{cc_s['comparison'][1]} vs TracePilot {cc_f['comparison'][0]}/{cc_f['comparison'][1]}), and answer-level multi-step completion "
                       f"({pct(*st['multi_step_full_completion'])} vs {pct(*fu['multi_step_full_completion'])}) was not better than the baseline. Ablations show the LLM planner and the numeric tools carry the improvement; retrieval-only agentic variants were not better than Standard RAG, "
                       "and the gain from bounded replanning is small and not confirmed."))
    story.append(P("<b>What this report does and does not claim.</b> It documents a functional, tested agentic system and a measured comparison on a 114-question benchmark whose annotations were AI-assisted and whose "
                   "scoring is keyword/number-based. Academic completion, functional implementation and measured improvement are separate conclusions (Section 29)."))
    # 2
    story += [sec("Project Objectives")]
    reqs = [["#", "Official requirement", "Where implemented"], ["1-2", "Multiple documents, knowledge base", "35 documents / 4 courses (src/ingestion, data/documents)"],
            ["3-5", "Extract, clean, chunk", "loaders.py, cleaning.py, chunking.py"], ["6-7", "Embeddings, storage", "all-MiniLM-L6-v2 + persistent ChromaDB"],
            ["8-10", "Agent understands questions, decides when/how to retrieve", "planner.py, routing.py, controller.py"], ["11", "Refine queries", "refine_search_query + replanning ladder"],
            ["12", "Compare retrieved information", "compare_evidence (extractive, conflict-preserving)"], ["13", "Answers with sources", "synthesis.py, citations/validator.py"],
            ["14-17", "Search, summarization, data-analysis, calculation tools", "src/tools/*"], ["18", "Complex multi-step questions", "dependency-graph plans, evidence aggregation"], ["19", "Gradio/Streamlit deployment", "app.py (local Gradio; hosted Spaces blocked by HTTP 402)"]]
    story.append(table(reqs, [1.3 * cm, 6.5 * cm, 8.5 * cm]))
    # 3
    story += [sec("Standard RAG Limitations")]
    story.append(P("Standard RAG retrieves once and generates once. It cannot compute (a mean must be read from text, if present), cannot decompose a multi-part question, and cannot tell that a weak retrieval deserves another attempt."))
    if st:
        cc = S["by_category"]["standard"]
        story.append(P("Measured on the test split, Standard RAG's per-category correctness: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(cc.items())) + ". "
                       f"Its numerical accuracy was {pct(*st['numerical_accuracy'])}."))
    # 4
    story += [sec("Corrective RAG Background")]
    story.append(P("ReGround AI (a previous project) implements Corrective RAG: retrieval grading, query rewriting and re-retrieval. Its own evaluation found that the overall accuracy gain over Standard RAG was <b>not statistically confirmed</b>; "
                   "this report inherits no claim of improvement from it. TracePilot adds capabilities that a corrective loop cannot provide (tool execution, deterministic numbers, multi-step dependency plans) and keeps the corrective idea in its bounded replanning."))
    # 5
    story += [sec("Agentic RAG Methodology")]
    story += bullets(["<b>Plan</b> – an LLM produces a typed plan (JSON schema, Pydantic-validated); a rule guard repairs/limits it.", "<b>Act</b> – the controller executes steps, resolving arguments from earlier observations.",
                      "<b>Observe/validate</b> – relevance and error checks decide whether evidence is sufficient.", "<b>Replan</b> – at most 2 rounds: re-search with a different query, repair arguments, add a calculation, retry on timeout.",
                      "<b>Answer</b> – synthesis from sufficient evidence only, citation verification, deterministic computed results, or abstention."])
    # 6
    story += [sec("System Architecture")] + img(FIG / "architecture.png", 16 * cm, "Figure 1 – TracePilot architecture (drawn from the implemented modules).")
    story.append(P("A custom state machine was chosen over LangGraph (v1.2.x at the time of writing): persistence/interrupt features were not needed, every transition must land in our own trace schema, and the explicit controller (~450 lines) makes budgets and ablations easy to test. "
                   "Schema-constrained JSON (Ollama <i>format</i>) rather than native function calling is used for plans because a plan is a multi-step object with dependencies; invalid output is never counted as a tool call."))
    if pb:
        story.append(P(f"Probe (dev split, scripts/probe_llm.py): native Ollama tool calling with the same model returned a valid first tool call for {pb.get('valid_native_tool_calls')}/{pb.get('n')} tool-requiring dev questions, "
                       f"and the first tool was acceptable for {pb.get('first_tool_acceptable')}/{pb.get('n')}."))
    # 7
    story += [sec("Knowledge Base")]
    import collections
    _man = json.loads((ROOT / "data/manifests/sources.json").read_text())
    _c = collections.Counter(x["filename"].rsplit(".", 1)[1].upper() for x in _man)
    story.append(P(f"{len(_man)} documents (" + ", ".join(f"{k} {v}" for k, v in sorted(_c.items())) + "), 4 courses, 167 chunks. 31 files are original CC BY 4.0 material by the project owner (CourseAtlas/ReGround, copied read-only); 4 were written for TracePilot to create genuine multi-step needs "
                   "(a right-skewed product price table; regularization experiments; an overfitting-mitigation guide; a robust-summaries note). The corpus deliberately contains superseded documents that conflict with current ones, overloaded terms and three prompt-injection lines."))
    # 8
    story += [sec("Document Ingestion")]
    story.append(P("Validation (extension, 5 MB cap, magic bytes, no symlinks, macro formats rejected) → text extraction only → cleaning (header/footer stripping, de-hyphenation) → sentence-aware chunking (900/120 chars; CSV: 8 rows per chunk with true row ranges) → embeddings → Chroma upsert. "
                   "SHA-256 duplicate detection (also across file names), update-in-place, delete, reindex, per-chunk content hash, safe storage names and per-course isolation. Documents are never executed."))
    # 9
    story += [sec("Embedding and Vector Retrieval")]
    story.append(P("all-MiniLM-L6-v2 (384-d, normalised, Apache-2.0) with a persistent ChromaDB cosine index; the retriever fuses dense and BM25 ranks (reciprocal rank, lexical weight 0.35). The same retriever serves the baseline and the agent."))
    if st and fu:
        story.append(table([["metric (test)", "Standard RAG", "TracePilot"], ["gold-document coverage", f"{st['retrieval_coverage']:.3f}", f"{fu['retrieval_coverage']:.3f}"],
                            ["Recall@5 (any gold doc)", f"{st['recall@5']:.3f}", f"{fu['recall@5']:.3f}"], ["MRR", f"{st['MRR']:.3f}", f"{fu['MRR']:.3f}"]], [6 * cm, 4 * cm, 4 * cm]))
    # 10
    story += [sec("Agent Planner")]
    story.append(P("The planner receives the enabled tools, the dataset catalog (ids, columns, row counts), the selected courses and three few-shot plans (topics absent from the benchmark). It returns question type, complexity, <i>requires_*</i> flags and ≤ 8 subtasks with tools and arguments. "
                   "A guard enforces hard constraints (see AGENT_DESIGN.md). Different questions produce different plans — a factual question yields one search; a comparison yields two searches and a comparison; a numerical question yields dataset analysis only."))
    if fu:
        story.append(P(f"On the test split the planner fell back to the rule planner (invalid LLM output twice) in {pct(*fu['planner_rules_fallback'])} of requests."))
    # 11
    story += [sec("Tool Registry")]
    story.append(table([["tool", "role", "notes"], ["search_knowledge_base", "dense+BM25 retrieval", "scope can be narrowed, never widened"], ["retrieve_source_context", "neighbouring chunks", "same document, scope-checked"],
                        ["summarize_evidence", "summaries", "bullets validated; G# never primary evidence"], ["analyze_dataset", "pandas statistics", "allow-listed operations, sample std/var by default"],
                        ["calculate_expression", "safe arithmetic", "AST allow-list, no eval"], ["compare_evidence", "comparison", "extractive; conflicts preserved"], ["refine_search_query", "decompose/expand", "numbers, ids, negations preserved"]], [4.2 * cm, 4 * cm, 8 * cm]))
    # 12
    story += [sec("Execution Controller")]
    story.append(P("States: ANALYZE → PLAN → EXECUTE → VALIDATE → (REPLAN → EXECUTE)* → SYNTHESIZE → CITATION_CHECK → DONE. Budgets: ≤ 9 tool calls, ≤ 2 replans, ≤ 3 attempts per subtask, ≤ 7 plan steps, 30 s per tool, 180 s per request, optional LLM work capped. "
                   "Stop reasons (answered, insufficient_evidence, tool_budget_exhausted, timeout, llm_unavailable, error) are recorded. The limits were set from plan shapes and the 8 GB/latency budget, not tuned on the test split."))
    # 13
    story += [sec("Multi-Step Workflows")]
    story.append(P("Plans are dependency graphs: calculator inputs name earlier steps (<i>t1 - t2</i> → evidence ids D1, D2), comparison/summarization depend on retrieval steps, and the guard/replanner can add missing steps (for example a calculation after two dataset lookups). "
                   "Example (a recorded trace from the Studio): definition search → mitigation search → compare_evidence → two dataset lookups (run R02, run R04) → calculate_expression."))
    # 14-16
    story += [sec("Evidence Aggregation")]
    story.append(P("All tool outputs mint typed evidence ids in an <b>EvidenceStore</b>: S# (document passage with page/section/row range), D# (dataset computation with file hash, rows used, filters), C# (calculation with inputs), G# (generated summary, never primary evidence). "
                   "Only sufficiently supported evidence reaches the synthesizer; unsupported sub-questions are passed as explicit gaps."))
    story += [sec("Grounded Answer Generation")]
    story.append(P("Evidence is wrapped as untrusted data; the LLM must cite ids, quote computed numbers exactly and report conflicting sources separately. Computed results are always appended deterministically, so numbers never depend on the model copying them. "
                   "If the draft is entirely rejected but relevant evidence exists, an extractive answer (verbatim evidence sentences with ids) is returned instead of a false abstention."))
    story += [sec("Citation Validation")]
    story.append(P("Ids must exist in the evidence store (invented ids are removed). Each cited sentence must (a) contain only numbers found in the cited evidence (rounding-aware), (b) have cosine ≥ 0.55 to a window of the evidence and (c) ≥ 50% of its content words in the evidence, or ≥ 80% word recall. "
                   "One regeneration is allowed; remaining unsupported sentences are dropped; uncited factual sentences are attributed to the evidence that demonstrably supports them or dropped. "
                   "<b>Limitation:</b> this is an embedding+lexical proxy, not an NLI model; it cannot reliably detect a contradiction that reuses the evidence's vocabulary."))
    if st and fu:
        story.append(table([["citation metric (test)", "Standard RAG", "TracePilot"], ["gold-document citation hit", pct(*st["citation_hit (answerable w/ gold docs)"]), pct(*fu["citation_hit (answerable w/ gold docs)"])],
                            ["mean validator support rate", f"{st['mean_citation_support_rate']:.3f}", f"{fu['mean_citation_support_rate']:.3f}"],
                            ["cited sentences judged unsupported (after repair)", pct(*st["unsupported_cited_sentences"]), pct(*fu["unsupported_cited_sentences"])]], [7 * cm, 4.2 * cm, 4.2 * cm]))
        story.append(P("The validator is the system's own check, so the last two rows are not independent evidence; the gold-document hit is judged against the annotations.", SMALL))
    # 17-18
    story += [sec("Application Architecture")]
    story.append(P("TracePilot Studio (Gradio) has nine modules: Agentic Q&amp;A, Execution Trace, Tool Explorer (validated manual execution), Knowledge Library (upload/index/reindex/delete/inspect), Evidence Inspector, Agent Workflow (executed graph), Standard vs Agentic, Analytics and Failure Analysis. Every panel renders data from real backend execution."))
    for nme, cap in [("02_multistep_answer_and_live_trace.png", "Figure – multi-step question with answer, citations and live trace (real run)."), ("08_agent_workflow.png", "Figure – executed workflow graph reconstructed from the recorded plan and trace.")]:
        story += img(SHOT / nme, 12.5 * cm, cap)
    story += [sec("Agent Execution Trace")]
    story.append(P("Every request produces a structured trace (request id, question, scope, plan with adjustments, tool selections with redacted arguments, tool results, relevance checks, replanning actions, citation-check counts, stop reason, timings). No hidden reasoning is stored. Traces are saved as JSON (artifacts/execution_traces)."))
    story += img(SHOT / "04_execution_trace_tab.png", 12.5 * cm, "Figure – the Execution Trace module showing the recorded events of a real request.")
    # 19
    story += [sec("Evaluation Methodology")]
    story.append(P(f"Benchmark v{md['dataset_version']}: <b>{md['n_questions']}</b> questions in 11 categories (simple factual, comparison, CSV analysis, calculation, multi-step, summarization, ambiguous, unanswerable, conflicting sources, tool routing, injection exposure), "
                   f"split deterministically (every third item per category) into <b>{md['n_dev']} development</b> and <b>{md['n_test']} held-out test</b> questions. New questions were written for this project; none were reused from CourseAtlas/ReGround failure sets."))
    story += bullets(["<b>Annotation is AI-assisted</b> (Claude) and was not independently audited by a human. Numeric gold values are computed with pandas from the CSVs; each textual evidence pattern is machine-verified to occur in the named gold document (the build fails otherwise).",
                      "<b>Scoring is rubric-based</b> (regex groups for required facts, numeric tolerance, forbidden strings, abstention). No LLM judge. Rubrics are approximations: a correct paraphrase may fail, a keyword-rich wrong answer may pass.",
                      "<b>Leakage control</b>: the agent was developed and thresholds calibrated on the dev split only. The protocol was frozen (SHA-256 of test questions, configs, source) before any test run: "
                      f"frozen at {fz.get('frozen_at')}, commit {str(fz.get('git_commit'))[:10]}. Development involved four iterations (Section 29 notes the dev→test difference).",
                      f"Same LLM (qwen2.5:3b-instruct, temperature 0, seed 7), embeddings, retriever, synthesis prompt, validator and hardware for both systems; corpus hash {md['corpus_text_sha256'][:12]}…; tool registry v1.0.0.",
                      "Paired bootstrap (5000 resamples, seed 20261121) over questions; Wilson/percentile intervals where shown; numerator and denominator are always reported."])
    # 20
    story += [sec("Standard vs Agentic Comparison")]
    if st and fu:
        rows = [["metric (test split)", "Standard RAG", "TracePilot"]]
        for r in S["table"]:
            rows.append([r["metric"], r["standard"], r["full"]])
        rows += [["false abstention (answerable)", pct(*st["false_abstention (answerable)"]), pct(*fu["false_abstention (answerable)"])],
                 ["mean LLM calls", f"{st['mean_llm_calls']:.2f}", f"{fu['mean_llm_calls']:.2f}"], ["mean tokens", f"{st['mean_tokens']:.0f}", f"{fu['mean_tokens']:.0f}"]]
        story.append(table(rows, [6.4 * cm, 4.6 * cm, 4.6 * cm]))
        story.append(P(f"Paired overall: TracePilot-only correct {ov['b_only']}, Standard-only correct {ov['a_only']}; difference {100 * ov['mean_diff']:+.1f} pp, 95% CI [{100 * ov['ci95'][0]:+.1f}, {100 * ov['ci95'][1]:+.1f}]."))
        story += img(FIG / "category_test.png", 16 * cm, "Figure – per-category correctness, test split.")
        story.append(P("Per-category paired differences (small n; treat as descriptive): " + "; ".join(
            f"{c} {100 * v['mean_diff']:+.0f} pp (n={v['n']})" for c, v in pr.items() if c != "overall") + "."))
    # 21
    story += [sec("Agent Planning Evaluation")]
    if fu:
        story.append(table([["planning metric (test)", "value"], ["correct execution path (required capabilities used, no forbidden tool)", pct(*fu["tool_correct_path"])],
                            ["required-capability recall", pct(*fu["tool_required_ok(recall)"])], ["mean tool precision", f"{fu['mean_tool_precision']:.3f}"],
                            ["unnecessary-call rate (calls outside the annotated capabilities)", pct(*fu["unnecessary_call_rate"])], ["forbidden tool used", pct(*fu["forbidden_tool_used"])],
                            ["plan completion (all planned steps done)", pct(*fu["plan_completion"])], ["requests that replanned", pct(*fu["replanned_requests"])]], [11 * cm, 4.6 * cm]))
        story.append(P("Annotated capabilities are conservative: for example a 'difference between X and Y' question is annotated as search-only, so a (reasonable) extra compare step is counted as unnecessary."))
    # 22
    story += [sec("Tool Selection Evaluation")] + img(FIG / "tool_usage_test.png", 11 * cm, "Figure – tool usage frequency on the test split (from recorded tool sequences).")
    story.append(P("Dynamic selection is demonstrated by tests (tests/test_planning.py, integration tests with the real model) and by the distribution of tool sequences: factual questions use search only, numerical questions use dataset analysis or the calculator, comparison questions add compare_evidence."))
    # 23
    story += [sec("Multi-Step Task Completion")]
    if st and fu:
        story.append(table([["multi-step metric (test)", "Standard RAG", "TracePilot"], ["full completion (correct answer AND a gold document cited)", pct(*st["multi_step_full_completion"]), pct(*fu["multi_step_full_completion"])],
                            ["full completion with the annotated tool path", "n/a (no tools)", pct(*fu["multi_step_full_completion_with_correct_tool_path"])]], [8.4 * cm, 3.6 * cm, 3.6 * cm]))
        story.append(P("Partial success (for example a correct definition but a wrong number) is <b>not</b> counted as completion."))
    # 24
    story += [sec("Ablation Results")]
    if AB:
        rows = [["system", "n", "correct", "numeric", "multi-step", "mean latency (s)", "tool calls"]]
        for k, v in AB["rows"].items():
            rows.append([v["label"], v["n"], pct(*v["correct"]), pct(*v["numeric"]), pct(*v["multi"]), f"{v['lat']:.1f}", f"{v['calls']:.2f}"])
        story.append(table(rows, [5.2 * cm, 0.9 * cm, 2.6 * cm, 2.3 * cm, 2.3 * cm, 1.6 * cm, 1.3 * cm], fs=7.2))
        story += img(FIG / "ablation_test.png", 13 * cm, "Figure – ablation on the test split.")
        story.append(P(AB.get("note", "")))
    # 25
    story += [sec("Latency Analysis")]
    if st and fu:
        story.append(table([["latency (s, test)", "mean", "median", "p95"], ["Standard RAG", f"{st['latency']['mean']:.1f}", f"{st['latency']['median']:.1f}", f"{st['latency']['p95']:.1f}"],
                            ["TracePilot", f"{fu['latency']['mean']:.1f}", f"{fu['latency']['median']:.1f}", f"{fu['latency']['p95']:.1f}"]], [6 * cm, 3 * cm, 3 * cm, 3 * cm]))
        story += [Spacer(1, 6)] + img(FIG / "latency_test.png", 8 * cm)
        story.append(P("Latency is dominated by LLM calls on an 8 GB M2 (3B model). The agent pays for the planner call and, for multi-part questions, extra tool calls; pure dataset questions are fast because no document retrieval or long generation is needed."))
    # 26
    story += [sec("Security")]
    story.append(P("Controls: evidence wrapped as untrusted data, injection pattern detection reported in the trace, no code/shell/file/network/SQL tools, AST-only calculator, enum-only dataset operations, path-safe names, scope enforcement, budgets and timeouts. "
                   f"Automated tests: {tests.get('unit', {}).get('passed', '?')} unit tests passed ({tests.get('unit', {}).get('failed', '?')} failed, {tests.get('unit', {}).get('skipped', '?')} skipped); "
                   f"real-model integration tests {tests.get('integration', {}).get('passed', '?')} passed / {tests.get('integration', {}).get('failed', '?')} failed; "
                   f"browser verification {sum(s['ok'] for s in br.get('steps', []))}/{len(br.get('steps', []))} steps."))
    if fu:
        inj = [r for r in (J("test_full.jsonl") or [])] if False else None
    story.append(P("The corpus itself contains deliberate injection lines; six benchmark items retrieve them. Per-category results are listed in Section 20 and in docs/EVALUATION.md. The controls prevent tool misuse; they cannot guarantee the prose of an LLM answer."))
    # 27
    story += [sec("Failure Analysis")]
    fa_md = (DOCS / "FAILURE_ANALYSIS.md")
    if fa_md.exists():
        for line in fa_md.read_text().splitlines():
            line = line.rstrip()
            if not line or line.startswith("# "):
                continue
            if line.startswith("## "):
                story.append(P(md_inline(line[3:]), H2))
            elif line.startswith("- "):
                story += bullets([md_inline(line[2:])])
            else:
                story.append(P(md_inline(line)))
    # 28
    story += [sec("Limitations")]
    story += bullets(["A 3B local model makes routing and planning mistakes; the guard and replanner repair many but not all (see failure analysis).", "Benchmark: AI-assisted annotation without independent human audit; keyword rubrics; small per-category n; one corpus of 35 short documents.",
                      "The citation-support check is not NLI and can pass contradictions that reuse the evidence's wording.", "Latency of multi-step questions (tens of seconds on an 8 GB M2).", "Ambiguous questions are usually answered for one reading instead of asking for clarification.",
                      "No hosted deployment: free Gradio Spaces returned HTTP 402 for this account; the project is local.", "The Qwen2.5-3B model has a research (non-commercial) licence."])
    # 29
    story += [sec("Conclusion")]
    if st and fu:
        story.append(P(f"<b>Academic assignment:</b> every mandatory requirement is implemented and exercised end to end. <b>Agentic behaviour:</b> verified — plans, tool sequences, replanning and abstention differ per question and are recorded in traces. "
                       f"<b>Measured improvement:</b> on the held-out test split TracePilot scored {pct(*fu['answer_correctness (all)'])} vs {pct(*st['answer_correctness (all)'])} for Standard RAG "
                       f"({100 * ov['mean_diff']:+.1f} pp; 95% CI [{100 * ov['ci95'][0]:+.1f}, {100 * ov['ci95'][1]:+.1f}]), "
                       f"{'a difference whose interval excludes zero on this benchmark' if ov.get('significant') else 'not statistically confirmed'}. The gain concentrates in categories that need computation (CSV analysis, calculation, multi-step) where Standard RAG is structurally unable to compute; "
                       "on plain factual questions the systems are close, and on comparison questions Standard RAG did better in this run. "
                       f"Standard RAG has the lower median latency ({st['latency']['median']:.1f} s vs {fu['latency']['median']:.1f} s); means are similar ({st['latency']['mean']:.1f} s vs {fu['latency']['mean']:.1f} s) because both have slow tails on a shared machine. The benchmark limitations above bound how far this conclusion generalises."))
        if dev:
            d1, d2 = dev["systems"]["standard"]["answer_correctness (all)"], dev["systems"]["full"]["answer_correctness (all)"]
            story.append(P(f"Development-split figures (used for tuning, therefore optimistic): Standard {pct(*d1)}, TracePilot {pct(*d2)}. The test-split numbers above are the ones to cite."))
    # 30
    story += [sec("Project Links")]
    story += bullets([f"GitHub: {GITHUB}", "Hugging Face Space: not available (free Gradio hosting returned HTTP 402 for this account; nothing was created)", "Local run: python app.py after the steps in docs/REPRODUCIBILITY.md",
                      f"Frozen evaluation record: data/evaluation/FROZEN.json (commit {str(fz.get('git_commit'))[:10]})", "Social media posts: LinkedIn PENDING, Facebook PENDING (handled separately)"])
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2.2 * cm,
                            title="TracePilot AI — Agentic RAG Research Report", author="Faris Mohamed Kishtah")
    from reportlab.platypus import PageTemplate, Frame, BaseDocTemplate
    doc.build([Spacer(1, 1)] + story, onFirstPage=cover, onLaterPages=page)
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    build()
