"""Query refinement: decomposition, terminology normalisation, synonym expansion - with validation that
numbers, identifiers, negations and quoted text survive. No LLM is needed; rewriting cannot invent facts."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.tools.base import Tool, ToolContext

ABBR = {"sd": "standard deviation", "std": "standard deviation", "knn": "k-nearest neighbors", "mse": "mean squared error", "cv": "cross-validation", "pca": "principal component analysis",
        "nlp": "natural language processing", "df": "DataFrame", "regularisation": "regularization"}
SYN = {"average": "mean", "overfit": "overfitting", "overfits": "overfitting", "avg": "mean",
       "variance": "variance spread", "outlier": "outlier extreme value", "outliers": "outliers extreme values",
       "skew": "skewed distribution", "skewed": "skewed distribution", "missing": "missing values null NaN",
       "mitigate": "reduce prevent", "reduce": "reduce mitigate", "supervised": "supervised learning labels"}
LEAD = re.compile(r"^\s*(please\s+)?(explain|define|describe|calculate|compute|compare|summari[sz]e|find|retrieve|give me|tell me|what is|what are|how (do|does|can|to)|why (do|does|is|are))\b[\s,:]*", re.I)
SPLIT = re.compile(r"\s*(?:,\s*then\s+|;\s*|\.\s+(?=[A-Z])|\band then\b|,\s*and\s+(?=(?:explain|compare|summari[sz]e|calculate|find|retrieve|compute|describe|define)\b)|\bthen\b)\s*", re.I)
NEG = re.compile(r"\b(not|no|never|without|except|excluding|neither|nor|non)\b", re.I)
NUMS = re.compile(r"\d+(?:\.\d+)?")
IDS = re.compile(r"\b[A-Z]{1,3}\d{2,4}\b|\b[a-z]+_[a-z_0-9]+\b|\"[^\"]+\"|'[^']+'")
STOPW = set("the a an of and or to in is are was were be what how why which when does do for on with by as at from that this it its between about into than then me you i my your".split())


def keywords(t: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9_+.-]+", t) if w.lower() not in STOPW and len(w) > 1]


VERBS = r"(?:explain|define|describe|calculate|compute|compare|contrast|summari[sz]e|find|retrieve|list|give me|tell me|show|analy[sz]e)"
CLAUSE = re.compile(r"\s*(?:[;,]\s*(?:and\s+)?(?:then\s+)?|\band then\b|\bthen\b|\band\s+(?=" + VERBS + r"\b))\s*", re.I)
PRON = {"them", "it", "these", "those", "this", "that", "both"}


def _strip_lead(p: str) -> str:
    p = LEAD.sub("", p).strip(" ,")
    return re.sub(r"^(the|their|its|a|an)\s+", "", p, flags=re.I)


def decompose(question: str) -> list[str]:
    q = question.strip().rstrip("?.! ")
    out = []
    for part in [p for p in CLAUSE.split(q) if p and p.strip()]:
        m = re.match(r"^\s*(?:compare|contrast)\s+(.+?)\s+(?:and|with|versus|vs\.?|to)\s+(.+)$", part, re.I)
        if m:
            x, y = _strip_lead(m.group(1)), _strip_lead(m.group(2))
            if len(x.split()) == 1 and len(y.split()) >= 2:       # "average and median price" -> shared head noun
                x = f"{x} {re.split(r'\s+(?:in|of|for|on|from|within)\s+', y)[0].split()[-1]}"
            out += [x, y]
        else:
            out.append(_strip_lead(part))
    seen, res = set(), []
    for o in out:
        if o and o.lower() not in PRON and len(o) > 2 and o.lower() not in seen:
            seen.add(o.lower())
            res.append(o)
    return res or [q]


def normalise(q: str) -> str:
    toks = []
    for w in q.split():
        core = re.sub(r"[^\w+-]", "", w.lower())
        toks.append(ABBR.get(core, w))
    return " ".join(toks)


def expand(q: str) -> str:
    extra = []
    for w in re.findall(r"[a-z]+", q.lower()):
        if w in SYN:
            extra += [t for t in SYN[w].split() if t.lower() not in q.lower()]
    return (q + " " + " ".join(dict.fromkeys(extra))).strip() if extra else q


def preserved(original: str, refined: list[str]) -> tuple[bool, list[str]]:
    """Every number / identifier / negation in the original must appear in some refined query; no new numbers may appear."""
    joined = " ".join(refined).lower()
    problems = []
    for n in NUMS.findall(original):
        if n not in joined:
            problems.append(f"number {n} dropped")
    for n in IDS.findall(original):
        if n.lower().strip("\"'") not in joined:
            problems.append(f"identifier {n} dropped")
    for m in NEG.findall(original):
        if m.lower() not in joined:
            problems.append(f"negation '{m}' dropped")
    for n in NUMS.findall(" ".join(refined)):
        if n not in original:
            problems.append(f"number {n} invented")
    return (not problems), problems


class RefineInput(BaseModel):
    question: str = Field(min_length=3, max_length=600)
    previous_queries: list[str] = Field(default_factory=list, max_length=6)
    mode: str = Field(default="auto", pattern="^(auto|decompose|expand|keywords)$")


def refine(question: str, previous: list[str], mode: str = "auto") -> tuple[list[str], list[str], bool, list[str]]:
    changes = []
    if mode in ("auto", "decompose"):
        subs = decompose(question)
        if len(subs) > 1:
            changes.append(f"decomposed into {len(subs)} sub-queries")
    else:
        subs = [LEAD.sub("", question.strip().rstrip("?.! ")).strip() or question]
    out = []
    for s in subs:
        n = normalise(s)
        if n != s:
            changes.append("normalised abbreviations")
        e = expand(n) if mode in ("auto", "expand") else n
        if e != n:
            changes.append("expanded synonyms")
        if mode == "keywords":
            e = " ".join(keywords(e))
            changes.append("reduced to keywords")
        out.append(e)
    prev = {p.lower() for p in previous}
    out = [o for o in dict.fromkeys(out) if o.lower() not in prev] or [expand(" ".join(keywords(question)))]
    ok, problems = preserved(question, out)
    if not ok:                                   # reject a rewrite that loses constraints
        return [question], ["rewrite rejected: " + "; ".join(problems)], False, problems
    return out[:4], sorted(set(changes)), True, []


class RefineSearchQuery(Tool):
    name = "refine_search_query"
    description = ("Improve retrieval for complex questions: split multi-part questions into sub-queries, normalise "
                   "abbreviations, expand synonyms. Numbers, identifiers and negations are always preserved.")
    Input = RefineInput
    output_description = "queries: list[str] (<=4), changes, validated flag"
    examples = [{"question": "Explain overfitting, compare L2 and early stopping, then summarize advantages"}]

    def run(self, a: RefineInput, ctx: ToolContext):
        qs, ch, ok, prob = refine(a.question, a.previous_queries, a.mode)
        return {"queries": qs, "changes": ch, "validated": ok, "rejected_reasons": prob}, [], f"{len(qs)} queries; " + ("; ".join(ch) or "unchanged")
