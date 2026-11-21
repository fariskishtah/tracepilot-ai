"""Citation validation: IDs must exist AND the cited evidence must support the sentence.
Syntax alone is never trusted."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from src.embeddings.embedding_service import embed

CITE = re.compile(r"\[((?:[SDCG]\d+)(?:\s*[,;]\s*[SDCG]\d+)*)\]")
NUM = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?%?")
STOP = set("the a an of and or to in is are was were be it its this that for on with by as at from which can than then also not".split())
DISCOURSE = set("should used use using under following shows show shown based provided mentioned according described indicates indicate refers refer states state means mean plugging value values given each such these those there their they when where while does only just into over more most other another".split())
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(*-])|\n+")


def ids_in(text: str) -> list[str]:
    out = []
    for m in CITE.finditer(text):
        out += [x.strip() for x in re.split(r"[,;]", m.group(1))]
    return out


def strip_cites(t: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", re.sub(CITE.pattern, " ", t)).replace(" .", ".").replace(" ,", ",").strip()


def parse_numbers(text: str) -> list[tuple[float, int]]:
    """(value, decimals) for every number token; percentages are kept as written."""
    out = []
    for n in NUM.findall(text):
        n = n.replace(",", "").rstrip("%")
        try:
            out.append((float(n), len(n.split(".")[1]) if "." in n else 0))
        except ValueError:
            pass
    return out


def norm_numbers(text: str) -> set[str]:
    return {f"{v:.6g}" for v, _ in parse_numbers(text)}


def numbers_supported(sentence: str, evidence_text: str, extra: set[str] | None = None) -> list[str]:
    """Numbers of the sentence that are NOT found in the evidence (allowing rounding to the
    sentence's own precision, and percent<->fraction forms). Small ordinals 1-5 are tolerated."""
    ev = [v for v, _ in parse_numbers(evidence_text)]
    missing = []
    for v, d in parse_numbers(sentence):
        if f"{v:.6g}" in (extra or set()) or (v == int(v) and 0 <= v <= 5):
            continue
        tol = 0.5 * 10 ** (-d) + 1e-9
        if any(abs(e - v) <= tol or abs(e * 100 - v) <= tol or abs(e / 100 - v) <= tol for e in ev):
            continue
        missing.append(f"{v:g}")
    return missing


def _words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP and len(w) > 2}


def _windows(text: str) -> list[str]:
    s = [x.strip() for x in SENT.split(text) if x.strip()]
    return s + [" ".join(s[i:i + 2]) for i in range(len(s) - 1)] or [text]


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[str]
    invalid_ids: list[str]
    supported: bool
    sim: float
    reason: str = ""


@dataclass
class ValidationReport:
    checks: list[SentenceCheck] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    uncited_numeric: list[str] = field(default_factory=list)

    @property
    def n_cited(self) -> int:
        return sum(1 for c in self.checks if c.cited)

    @property
    def n_supported(self) -> int:
        return sum(1 for c in self.checks if c.cited and c.supported)

    @property
    def support_rate(self) -> float:
        return self.n_supported / self.n_cited if self.n_cited else 0.0

    def unsupported(self) -> list[SentenceCheck]:
        return [c for c in self.checks if c.cited and not c.supported]

    def to_dict(self) -> dict:
        return {"n_cited_sentences": self.n_cited, "n_supported": self.n_supported,
                "support_rate": round(self.support_rate, 3), "invalid_ids": self.invalid_ids,
                "unsupported": [{"sentence": c.sentence[:200], "cited": c.cited, "reason": c.reason, "sim": round(c.sim, 2)} for c in self.unsupported()],
                "uncited_numeric_sentences": self.uncited_numeric}


def _stems(t: str) -> set[str]:
    return {w[:5] for w in _words(t) if w not in DISCOURSE}


def sentence_supported(sentence: str, ev_texts: list[str], sim_threshold: float = 0.6, extra_numbers: set[str] | None = None) -> tuple[bool, float, str]:
    """Support = (a) every number of the sentence appears in the cited evidence, AND
    (b) the sentence is semantically close to a window of the cited evidence (MiniLM cosine) AND most of its content
    words occur in that window (word recall). This is an approximate entailment proxy, NOT an NLI model: it catches
    off-topic or invented claims but cannot reliably detect contradictions that reuse the evidence's vocabulary."""
    s = strip_cites(sentence)
    if not ev_texts:
        return False, 0.0, "no evidence"
    ev_all = " ".join(ev_texts)
    missing = numbers_supported(s, ev_all, extra_numbers)
    if missing:
        return False, 0.0, f"numbers not in evidence: {missing[:3]}"
    wins = [w for t in ev_texts for w in _windows(t)]
    sw = _stems(s)
    recall = len(sw & _stems(ev_all)) / max(1, len(sw))        # words found anywhere in the cited evidence
    E = embed([s] + wins)
    sim = float(np.max(E[1:] @ E[0])) if len(wins) else 0.0
    ok = (sim >= 0.55 and recall >= 0.5) or recall >= 0.8
    return ok, sim, "" if ok else f"low support (cosine {sim:.2f}, word recall {recall:.2f})"


def validate_answer(answer: str, evidence: dict, question_numbers: set[str] | None = None) -> ValidationReport:
    rep = ValidationReport()
    for raw in SENT.split(answer):
        sent = raw.strip(" -*•\t")
        if len(strip_cites(sent)) < 8:
            continue
        cited = ids_in(sent)
        bad = [c for c in cited if c not in evidence]
        good = [c for c in cited if c in evidence]
        rep.invalid_ids += [b for b in bad if b not in rep.invalid_ids]
        if not cited:
            if numbers_supported(strip_cites(sent), '', question_numbers):
                rep.uncited_numeric.append(sent[:160])
            continue
        texts = [evidence[c].text for c in good]
        ok, sim, why = sentence_supported(sent, texts, extra_numbers=question_numbers) if good else (False, 0.0, "invalid citation id")
        rep.checks.append(SentenceCheck(sent, cited, bad, ok and not bad, sim, why or ("invalid id" if bad else "")))
    return rep
