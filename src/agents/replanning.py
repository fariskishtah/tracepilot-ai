"""Evidence sufficiency checks and bounded replanning actions (deterministic, traceable)."""
from __future__ import annotations

import difflib
import re

import pandas as pd

from src.agents.routing import OPS, analyze_question, mentioned_columns, pick_column, pick_dataset, rule_plan
from src.agents.state import AgentState, StepRecord, Subtask, SubtaskFilter
from src.config import cfg
from src.tools.query_refinement import keywords, refine

STOP = set("difference between explain define compare find evidence retrieve calculate summary summarize describe definition meaning "
           "two three using based".split())


def kw_coverage(query: str, texts: list[str]) -> float:
    kws = [k.lower() for k in keywords(query) if k.lower() not in STOP and len(k) > 2]
    if not kws:
        return 1.0
    toks = set(re.findall(r"[a-z0-9]+", " ".join(texts).lower()))
    stems = {t[:5] for t in toks}
    hit = sum(1 for k in kws if k in toks or (len(k) >= 5 and k[:5] in stems))
    return hit / len(kws)


def search_sufficient(query: str, data: dict) -> tuple[bool, dict]:
    sc = cfg("agent")["sufficiency"]
    hits = data.get("hits", [])
    best = data.get("best_score", 0.0)
    cov = kw_coverage(query, [h["text"] for h in hits[:3]])
    ok = bool(hits) and best >= sc["min_best_score"] and cov >= sc["min_keyword_coverage"]
    return ok, {"best_score": best, "keyword_coverage": round(cov, 2), "threshold": sc}


def value_lookup(text: str, ds: dict, kb) -> list[SubtaskFilter]:
    """Pick ONE row by lexical match between `text` and the row's non-numeric cells; filter on the first (id) column."""
    path = kb.dataset_path(ds["dataset_id"])
    if not path:
        return []
    df = pd.read_csv(path)
    toks = set(re.findall(r"[a-z0-9]+", text.lower()))
    idcol = df.columns[0]
    # direct id mention (e.g. ML004, O005)
    for v in df[idcol].astype(str):
        if v.lower() in toks:
            return [SubtaskFilter(column=idcol, op="==", value=v)]
    scores = []
    for i, row in df.iterrows():
        cells = " ".join(str(v) for c, v in row.items() if not pd.api.types.is_numeric_dtype(df[c]))
        rt = set(re.findall(r"[a-z0-9]+", cells.lower()))
        scores.append((len(rt & toks) / max(1, len(rt)), len(rt & toks), i))
    scores.sort(reverse=True)
    if scores and scores[0][1] >= 2 and (len(scores) == 1 or scores[0][0] > scores[1][0] + 0.05):
        return [SubtaskFilter(column=idcol, op="==", value=str(df.iloc[scores[0][2]][idcol]))]
    return []


def repair_analysis(rec: StepRecord, state: AgentState, kb, catalog: list[dict]) -> tuple[Subtask | None, str, list[Subtask]]:
    """Targeted fix of a failed analyze_dataset step from its error message."""
    s = rec.subtask.model_copy(deep=True)
    err = (rec.result.error or "") if rec.result else ""
    ds = next((d for d in catalog if d["dataset_id"] == s.dataset_id), None) or pick_dataset(state.question + " " + s.goal, catalog)
    if not ds:
        return None, "no dataset available", []
    s.dataset_id = ds["dataset_id"]
    changed = []
    if "not found" in err and "column" in err or s.column not in (ds["columns"] or []):
        col = pick_column(s.goal + " " + state.question, ds["columns"] or [])
        if not col and s.column:
            m = difflib.get_close_matches(s.column, ds["columns"] or [], n=1, cutoff=0.4)
            col = m[0] if m else None
        if col and col != s.column:
            s.column = col
            changed.append(f"column -> {col}")
    badf = [f for f in (s.filters or []) if f.column not in (ds["columns"] or [])]
    if badf:
        s.filters = value_lookup(s.goal, ds, kb) or None
        changed.append("invalid filter columns replaced" if s.filters else "invalid filters removed")
    if "exactly 1 matching row" in err or "no rows match" in err or (s.operation == "lookup_value" and not s.filters):
        f = value_lookup(s.goal, ds, kb) or value_lookup(state.question, ds, kb)
        if f:
            s.filters = f
            changed.append(f"filter -> {f[0].column}=={f[0].value}")
    if s.operation not in ("row_count", "describe_columns", "missing_values") and not s.column:
        op = next((o for p, o in OPS if re.search(p, state.question, re.I)), None)
        col = pick_column(state.question, ds["columns"] or [])
        if col:
            s.column = col
            s.operation = s.operation or op or "mean"
            changed.append(f"column -> {col}")
    extras: list[Subtask] = []
    named = mentioned_columns(s.goal, ds["columns"] or [])
    if s.operation == "lookup_value" and len(named) >= 2:          # one step asked for several columns -> one step per column
        s.column = named[0]
        extras = [s.model_copy(update={"column": c, "goal": f"{c} of the selected row"}) for c in named[1:3]]
        changed.append(f"split into {1 + len(extras)} lookups ({', '.join(named[:3])})")
    if not changed:
        return None, "no automatic repair available", []
    return s, "; ".join(changed), extras


def next_query(rec: StepRecord, question: str) -> tuple[str, str]:
    """A genuinely different query for a weak search (ladder: synonym expansion -> explanatory rephrase -> specific keywords)."""
    base = rec.subtask.query or rec.subtask.goal
    used = {x.lower() for x in rec.queries_used}
    cands: list[tuple[str, str]] = []
    qs, ch, ok, _ = refine(base, rec.queries_used, "expand")
    if ok and qs:
        cands.append((qs[0], "expand: " + ("; ".join(ch) or "synonyms")))
    kws = keywords(base)
    cands.append((f"definition and explanation of {' '.join(kws)}", "rephrase as explanation request"))
    spec = sorted(set(kws), key=len, reverse=True)[:4]
    cands.append((" ".join(spec), "most specific keywords only"))
    cands.append((f"{' '.join(keywords(question))} example formula notes", "question keywords + context words"))
    for q, how in cands:
        if q.strip() and q.lower() not in used:
            return q[:300], how
    return f"{base} overview", "fallback rephrase"
