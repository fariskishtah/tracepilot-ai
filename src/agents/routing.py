"""Question analysis (feature extraction), rule-based planning and the plan guard.

Rules are used (a) as the 'rules' planner in ablations, (b) as a fallback when the LLM planner returns
invalid output (recorded in the trace as plan_source=rules_fallback), and (c) by the guard to repair
plans that violate hard constraints (unknown/disabled tools, missing mandatory capability)."""
from __future__ import annotations

import re

from src.agents.state import QuestionPlan, Subtask, SubtaskFilter
from src.embeddings.embedding_service import embed
from src.tools.query_refinement import NUMS, decompose, keywords, normalise

CALC = re.compile(r"\b(calculate|compute|difference|average|mean|median|mode|standard deviation|variance|sum|total|how many|count|"
                  r"maximum|minimum|max|min|highest|lowest|correlat\w+|percent\w*|ratio|range|quantile|std|gap)\b", re.I)
DATA = re.compile(r"\b(csv|dataset|data set|table|column|columns|rows?|records?|runs?|experiment\w*|orders?|prices?|scores?|"
                  r"accuracy|accuracies|in the data|uploaded)\b", re.I)
COMPARE = re.compile(r"\b(compare|comparison|contrast|differ\w*|difference between|versus|vs\.?|similarities|better than|pros and cons|advantages)\b", re.I)
SUMM = re.compile(r"\b(summari[sz]e|summary|overview|key points|tl;?dr)\b", re.I)
MULTI = re.compile(r",\s*(?:then|and)\b|\bthen\b|;|\band (?:explain|compare|calculate|summari[sz]e|find|retrieve)\b", re.I)
NUMERIC_Q = re.compile(r"\b(calculate|compute|what is the (?:sample |population )?(?:mean|median|variance|standard deviation|sum|max|min)|how many rows|"
                       r"average|median|standard deviation|variance|correlation)\b", re.I)


from src.tools.query_refinement import CLAUSE as CLAUSE_SPLIT  # noqa: E402
GENERIC = {"explain", "describe", "define", "they", "them", "it", "why", "what", "how", "does", "do"}
EXPLAIN = re.compile(r"\b(why|explain|define|definition|describe|what (?:is|are)|how (?:does|do|can)|retrieve|identify|list|find the (?:definition|meaning|techniques?|methods?))\b", re.I)
OPS_ALLOWED = {"describe_columns", "row_count", "missing_values", "mean", "median", "min", "max", "sum", "std", "var", "count", "quantile", "value_counts", "groupby_agg", "correlation", "select_rows", "lookup_value"}
AGGS = {"mean", "median", "sum", "min", "max", "count", "std"}
DATAWORDS = re.compile(r"\b(csv|dataset|data set|table|column|rows?|experiment\w*|runs?)\b", re.I)


def analyze_question(q: str, catalog: list[dict]) -> dict:
    ids = [d["dataset_id"] for d in catalog]
    named = [i for i in ids if i.lower().replace("_", " ") in q.lower() or i.lower() in q.lower()]
    f = {
        "calc_cue": bool(CALC.search(q)), "data_cue": bool(DATA.search(q)) or bool(named),
        "compare_cue": bool(COMPARE.search(q)), "summary_cue": bool(SUMM.search(q)),
        "multi_cue": bool(MULTI.search(q)), "numbers_in_question": NUMS.findall(q), "named_datasets": named,
        "length_words": len(q.split()),
    }
    f["needs_data"] = f["calc_cue"] and f["data_cue"] and bool(NUMERIC_Q.search(q) or f["named_datasets"] or re.search(r"\b(csv|dataset|column|rows?)\b", q, re.I))
    cols = {c.lower() for d in catalog for c in (d["columns"] or [])}
    ql = q.lower()
    f["explicit_dataset"] = bool(re.search(r"\b(csv|dataset|data set|table|column|columns|rows?|spreadsheet)\b", q, re.I)) or bool(named) \
        or any(c in ql for c in cols if "_" in c) or any(i.lower().replace("_", " ") in ql for i in ids)
    f["literal_numbers"] = re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w])", q)
    f["needs_calc"] = len(f["literal_numbers"]) >= 2 and not f["explicit_dataset"] and bool(
        re.search(r"\b(calculate|compute|evaluate|how much|percent\w*|%|z-score|standard error|variance|mean of|average of|gap|increase|of)\b|[%*/+^]", q, re.I))
    f["needs_data"] = f["needs_data"] and f["explicit_dataset"]
    f["concept_cue"] = bool(re.search(r"\b(what|why|how|explain|define|describe|difference|compare|summari[sz]e|when|which)\b", q, re.I))
    return f


OPS = [("standard deviation|std|stdev", "std"), ("variance", "var"), ("median", "median"), ("average|mean", "mean"),
       ("maximum|highest|largest|\\bmax\\b|best", "max"), ("minimum|lowest|smallest|\\bmin\\b|worst", "min"),
       ("total|\\bsum\\b", "sum"), ("how many rows|number of rows|row count|how many records", "row_count"),
       ("correlat", "correlation"), ("missing", "missing_values")]
COLSYN = {"validation": "val", "val": "val", "training": "train", "price": "price", "prices": "price", "amount": "amount",
          "accuracy": "accuracy", "accuracies": "accuracy", "score": "score", "scores": "score", "epochs": "epochs"}


def pick_dataset(q: str, catalog: list[dict]) -> dict | None:
    if not catalog:
        return None
    for d in catalog:
        if d["dataset_id"].lower() in q.lower() or d["dataset_id"].lower().replace("_", " ") in q.lower():
            return d
    docs = [f"{d['dataset_id'].replace('_', ' ')} {' '.join(d['columns'] or [])}" for d in catalog]
    E = embed([q] + docs)
    lex = [len(set(re.findall(r"[a-z]+", q.lower())) & set(re.findall(r"[a-z]+", t.lower()))) for t in docs]
    sc = E[1:] @ E[0] + 0.05 * (E[1:] @ E[0] * 0 + lex)
    return catalog[int(sc.argmax())]


GENERIC_COLS = {"dataset", "model", "notes", "task", "technique", "note", "topic", "status", "category", "region", "section"}


def mentioned_columns(q: str, cols: list[str]) -> list[str]:
    """Columns named verbatim (e.g. 'train_accuracy') in the text, in order of appearance."""
    ql = q.lower()
    hit = [(ql.index(c.lower()), c) for c in cols if ("_" in c or len(c) > 5) and re.search(rf"(?<![a-z0-9_]){re.escape(c.lower())}(?![a-z0-9_])", ql)]
    return [c for _, c in sorted(hit)]


def pick_column(q: str, cols: list[str], numeric_only_hint=True) -> str | None:
    m = mentioned_columns(q, cols)
    if m:
        return m[0]
    ql = set(COLSYN.get(w, w) for w in re.findall(r"[a-z]+", q.lower()))
    best, bs = None, 0.0
    for c in cols:
        if c.lower() in GENERIC_COLS:
            continue
        parts = set(COLSYN.get(p, p) for p in re.split(r"[_\W]+", c.lower()) if p)
        sc = len(parts & ql) / max(1, len(parts))
        if "id" in parts and len(parts) == 1:
            sc = 0
        if sc > bs:
            best, bs = c, sc
    return best if bs >= 0.5 else None


def _filters(q: str, ds: dict, path_values) -> list[SubtaskFilter]:
    return path_values(q, ds)


def rule_plan(question: str, catalog: list[dict], f: dict, enabled: list[str], course_ids: list[str] | None, value_lookup=None) -> QuestionPlan:
    """Deterministic plan from question features. Different questions yield different tool sequences."""
    steps: list[Subtask] = []
    n = 0
    nid = lambda: f"t{(n := len(steps) + 1)}"
    subs = decompose(normalise(question))
    ds = pick_dataset(question, catalog) if f["needs_data"] and "analyze_dataset" in enabled else None
    analysis_ids: list[str] = []
    if ds:
        op = next((o for p, o in OPS if re.search(p, question, re.I)), "mean")
        col = pick_column(question, ds["columns"] or [])
        ops_in_q = [o for p, o in OPS if re.search(p, question, re.I)]
        multi_ops = [o for o in dict.fromkeys(ops_in_q) if o in ("mean", "median", "std", "var", "min", "max")][:2]
        flt = value_lookup(question, ds) if value_lookup else []
        if flt:
            multi_ops, op = [], "lookup_value"
        for o in (multi_ops if len(multi_ops) >= 2 else [op]):
            if o in ("row_count", "missing_values") or col:
                tid = nid()
                steps.append(Subtask(id=tid, tool="analyze_dataset", goal=f"{o} of {col or 'dataset'}", dataset_id=ds["dataset_id"],
                                     operation=o, column=col, filters=flt or None))
                analysis_ids.append(tid)
        if not analysis_ids:  # column unknown: inspect structure first (agentic exploration)
            tid = nid()
            steps.append(Subtask(id=tid, tool="analyze_dataset", goal="inspect columns", dataset_id=ds["dataset_id"], operation="describe_columns"))
            analysis_ids.append(tid)
    search_ids: list[str] = []
    need_concepts = f["concept_cue"] and (not ds or bool(re.search(r"\b(why|explain|define|definition|describe|compare|how (?:does|do|can))\b", question, re.I)))
    if "search_knowledge_base" in enabled and (need_concepts or not ds):
        qs = [s for s in subs if not (ds and re.search(r"\b(calculate|compute|difference|mean|median|average)\b", s, re.I) and not re.search(r"\b(why|explain)\b", s, re.I))] or subs
        if ds and len(qs) > 2:
            qs = qs[:2]
        qs = qs[:3] if not f["compare_cue"] else qs[:3]
        for s in qs:
            tid = nid()
            steps.append(Subtask(id=tid, tool="search_knowledge_base", goal=f"find evidence: {s}", query=s))
            search_ids.append(tid)
    if f["compare_cue"] and len(search_ids) >= 2 and "compare_evidence" in enabled:
        steps.append(Subtask(id=nid(), tool="compare_evidence", goal="compare retrieved evidence", depends_on=search_ids,
                             aspects=[re.sub(r"^find evidence: ", "", next(s.goal for s in steps if s.id == t)) for t in search_ids][:4]))
    if f["summary_cue"] and search_ids and "summarize_evidence" in enabled:
        steps.append(Subtask(id=nid(), tool="summarize_evidence", goal=question[:200], depends_on=search_ids))
    if len(analysis_ids) >= 2 and re.search(r"\b(difference|gap|ratio|percent\w*|change)\b", question, re.I) and "calculate_expression" in enabled:
        a, b = analysis_ids[0], analysis_ids[1]
        expr = f"{a} - {b}" if re.search(r"difference|gap", question, re.I) else f"pct_change({a}, {b})"
        steps.append(Subtask(id=nid(), tool="calculate_expression", goal="combine numeric results", expression=expr, depends_on=[a, b]))
    if not steps:
        steps.append(Subtask(id="t1", tool="search_knowledge_base", goal="find evidence", query=question[:300]))
    steps = steps[: 6]
    qt = ("multi_step" if len(steps) >= 4 or (ds and search_ids) else "data_analysis" if ds else "comparison" if f["compare_cue"]
          else "summarization" if f["summary_cue"] else "factual")
    return QuestionPlan(question_type=qt, complexity="simple" if len(steps) == 1 else "moderate" if len(steps) <= 3 else "complex",
                        requires_retrieval=bool(search_ids), requires_calculation=any(s.tool == "calculate_expression" for s in steps),
                        requires_data_analysis=bool(analysis_ids), requires_comparison=any(s.tool == "compare_evidence" for s in steps),
                        subtasks=steps, rationale="rule-based plan from question features")


def guard_plan(plan: QuestionPlan, question: str, f: dict, enabled: list[str], catalog: list[dict], max_steps: int) -> tuple[QuestionPlan, list[str]]:
    """Hard constraints. Returns the repaired plan and a list of human-readable adjustments (traced)."""
    adj: list[str] = []
    keep = []
    for s in plan.subtasks:
        if s.tool not in enabled:
            adj.append(f"removed {s.id}: tool '{s.tool}' unavailable")
            continue
        s.depends_on = [d for d in s.depends_on if d != s.id]
        keep.append(s)
    ids = {s.id for s in keep}
    for s in keep:
        s.depends_on = [d for d in s.depends_on if d in ids]
    # drop nonsensical dependents of removed steps, enforce DAG order (dependencies must come earlier)
    seen, ordered = set(), []
    for s in keep:
        if all(d in seen for d in s.depends_on):
            ordered.append(s)
            seen.add(s.id)
        else:
            adj.append(f"removed {s.id}: depends on a later/removed step")
    keep = ordered
    # tools that need upstream steps but have none -> attach all search steps
    searches = [s.id for s in keep if s.tool == "search_knowledge_base"]
    for s in keep:
        if s.tool in ("compare_evidence", "summarize_evidence", "retrieve_source_context") and not s.depends_on and searches:
            s.depends_on = [x for x in searches if x != s.id][:4]
            adj.append(f"{s.id}: attached dependencies {s.depends_on}")
    # mandatory capability: numeric/dataset question must have an analysis step if the tool exists
    if f["needs_data"] and "analyze_dataset" in enabled and not any(s.tool == "analyze_dataset" for s in keep) and catalog:
        ds = pick_dataset(question, catalog)
        n = len(keep) + 1
        keep.insert(0, Subtask(id=f"t{max([int(s.id[1:]) for s in keep] + [0]) + 1}", tool="analyze_dataset", goal="dataset analysis required by question",
                               dataset_id=ds["dataset_id"], operation="describe_columns"))
        adj.append("added analyze_dataset (question requires dataset numbers)")
    if not any(s.tool in ("search_knowledge_base", "analyze_dataset") for s in keep) and "search_knowledge_base" in enabled:
        keep.insert(0, Subtask(id=f"t{max([int(s.id[1:]) for s in keep] + [0]) + 1}", tool="search_knowledge_base", goal="find evidence", query=question[:300]))
        adj.append("added search (plan had no evidence-gathering step)")
    # analysis steps need an explicit dataset reference in the question (small LLMs over-use analyze_dataset)
    if not f.get("explicit_dataset"):
        for s_ in list(keep):
            if s_.tool == "analyze_dataset":
                keep.remove(s_)
                adj.append(f"removed {s_.id}: question does not refer to a dataset")
        ids_left = {x.id for x in keep}
        for s_ in list(keep):
            if s_.tool == "calculate_expression" and set(re.findall(r"\bt\d{1,2}\b", s_.expression or "")) - ids_left:
                keep.remove(s_)
                adj.append(f"removed {s_.id}: its inputs were removed")
    # compare/summarize only make sense over retrieved DOCUMENT sources
    sids = {s.id for s in keep if s.tool == "search_knowledge_base"}
    for s in list(keep):
        if s.tool in ("compare_evidence", "summarize_evidence", "retrieve_source_context"):
            s.depends_on = [d for d in s.depends_on if d in sids]
            if not s.depends_on:
                keep.remove(s)
                adj.append(f"removed {s.id}: {s.tool} has no retrieval step to work on")
    # explanatory clauses (why/explain/define/...) need retrieval even if the plan only computes numbers
    if "search_knowledge_base" in enabled:
        have = [s.query or s.goal for s in keep if s.tool == "search_knowledge_base"]
        clauses = [c for c in CLAUSE_SPLIT.split(question.rstrip("?.! ")) if c.strip()]
        for k, c in enumerate(clauses):
            if EXPLAIN.search(c) and not (DATAWORDS.search(c) and not re.search(r"\b(why|explain|define|definition|describe)\b", c, re.I)) and not any(len((set(keywords(c)) - GENERIC) & set(keywords(h + ' ' + g))) >= 1 for h, g in zip(have, [s.goal for s in keep if s.tool == 'search_knowledge_base'])) and len(keep) < max_steps:
                ctx = " ".join(keywords(clauses[k - 1])) if k else " ".join(keywords(question))
                qtxt = f"{re.sub(r'^(explain|describe)\\s+', '', c.strip(), flags=re.I)} {ctx}"[:200]
                nid = f"t{max([int(x.id[1:]) for x in keep] + [0]) + 1}"
                keep.append(Subtask(id=nid, tool="search_knowledge_base", goal=f"explain: {c.strip()[:100]}", query=qtxt))
                adj.append(f"added {nid}: question asks for an explanation that the plan did not retrieve")
                have.append(qtxt)
    # preflight: analysis steps that cannot possibly run (missing mandatory fields) are removed instead of wasting budget
    need_col = {"mean", "median", "min", "max", "sum", "std", "var", "quantile", "count", "value_counts", "groupby_agg", "correlation", "lookup_value"}
    for s in list(keep):
        if s.tool != "analyze_dataset":
            continue
        ds = next((d for d in catalog if d["dataset_id"] == s.dataset_id), None)
        cols = (ds or {}).get("columns") or []
        bad = (s.operation not in OPS_ALLOWED) or (s.operation in need_col and not (s.column in cols or pick_column(s.goal, cols))) \
            or (s.operation == "groupby_agg" and not s.group_by) or (s.agg and s.agg not in AGGS)
        if bad:
            keep.remove(s)
            adj.append(f"removed {s.id}: analyze_dataset step is not executable (operation/column/group_by invalid)")
    srch = [s.id for s in keep if s.tool == "search_knowledge_base"]
    if COMPARE.search(question) and len(srch) < 2 and "search_knowledge_base" in enabled and not f.get("needs_data"):
        parts = decompose(normalise(question))
        if len(parts) >= 2 and len(keep) + 2 <= max_steps:
            keep = [x for x in keep if x.tool != "search_knowledge_base"]
            for part in parts[:3]:
                nid = f"t{max([int(x.id[1:]) for x in keep] + [0]) + 1}"
                keep.append(Subtask(id=nid, tool="search_knowledge_base", goal=f"evidence on {part}", query=part))
            srch = [x.id for x in keep if x.tool == "search_knowledge_base"]
            adj.append("comparison question: decomposed into one search per side")
    if SUMM.search(question) and srch and "summarize_evidence" in enabled and not any(x.tool == "summarize_evidence" for x in keep) and len(keep) < max_steps:
        nid = f"t{max([int(x.id[1:]) for x in keep] + [0]) + 1}"
        keep.append(Subtask(id=nid, tool="summarize_evidence", goal=question[:200], depends_on=srch[:4]))
        adj.append("summarization question: added summarize_evidence")
    if COMPARE.search(question) and len(srch) >= 2 and "compare_evidence" in enabled and not any(s.tool == "compare_evidence" for s in keep) and len(keep) < max_steps:
        nid = f"t{max([int(x.id[1:]) for x in keep] + [0]) + 1}"
        keep.append(Subtask(id=nid, tool="compare_evidence", goal="compare retrieved evidence", depends_on=srch[:4]))
        adj.append(f"added {nid}: question asks for a comparison over retrieved sources")
    # redundant structure inspection when a real analysis of the same dataset is already planned
    real = {s.dataset_id for s in keep if s.tool == "analyze_dataset" and s.operation != "describe_columns"}
    for s in list(keep):
        if s.tool == "analyze_dataset" and s.operation == "describe_columns" and s.dataset_id in real:
            keep.remove(s)
            adj.append(f"removed {s.id}: redundant describe_columns")
    for s in keep:
        if s.tool == "calculate_expression":
            ids = {x.id for x in keep if x.tool in ("analyze_dataset", "calculate_expression")}
            names = set(re.findall(r"[A-Za-z_]\w*", s.expression or "")) - {"abs", "round", "sqrt", "log", "mean", "median", "pct_change", "percent_of", "pct_diff", "min", "max", "sum", "stdev", "var"}
            prior = [x.id for x in keep[: keep.index(s)] if x.tool in ("analyze_dataset", "calculate_expression") and x.operation != "describe_columns"]
            if not s.expression or not names <= ids:
                if len(prior) >= 2:
                    s.expression = f"{prior[0]} - {prior[1]}" if not re.search(r"percent|ratio|change", question, re.I) else f"pct_change({prior[1]}, {prior[0]})"
                    s.depends_on = prior[:2]
                    adj.append(f"{s.id}: repaired invalid expression variables")
                else:
                    keep.remove(s)
                    adj.append(f"removed {s.id}: expression references unavailable results")
            else:
                s.depends_on = sorted(names & ids)
    if len(keep) > max_steps:
        adj.append(f"truncated plan from {len(keep)} to {max_steps} steps")
        keep = keep[:max_steps]
    if not keep:  # nothing executable survived: fall back to the safest generic plan (retrieval over the question itself)
        if "search_knowledge_base" not in enabled:
            raise ValueError("empty plan after guard")
        keep = [Subtask(id="t1", tool="search_knowledge_base", goal="find evidence", query=question[:300])]
        adj.append("plan had no executable steps; replaced by a single search over the question")
    plan.subtasks = keep
    plan.requires_data_analysis = any(s.tool == "analyze_dataset" for s in keep)
    plan.requires_calculation = any(s.tool == "calculate_expression" for s in keep)
    plan.requires_comparison = any(s.tool == "compare_evidence" for s in keep)
    plan.requires_retrieval = any(s.tool == "search_knowledge_base" for s in keep)
    return plan, adj
