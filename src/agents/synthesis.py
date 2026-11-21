"""Grounded answer synthesis shared by the agent and the Standard-RAG baseline (identical prompt,
validator and post-processing, so the comparison is fair)."""
from __future__ import annotations

import re

from src.citations.validator import CITE, SENT, ids_in, norm_numbers, sentence_supported, strip_cites, validate_answer, ValidationReport
from src.config import cfg
from src.llm import LLMError
from src.security.guard import SYSTEM_SAFETY, wrap_untrusted, injection_flags
from src.tools.base import EvidenceItem

MARK = "INSUFFICIENT_EVIDENCE"
SYSTEM = SYSTEM_SAFETY + (
    " Answer the user's question using ONLY the supplied evidence and computed results. Rules: "
    "(1) Every sentence that states a fact must end with the citation ids of the evidence it uses, like [S1] or [S2][D1]; "
    "use only ids that appear in the evidence. (2) Quote computed numbers exactly as given and cite their [D#]/[C#] id. "
    "(3) If two sources disagree, report both positions with their ids; do not merge them. "
    "(4) Do not use outside knowledge. If part of the question is not covered, say that part is not covered. "
    f"(5) If the evidence cannot answer the question at all, reply with exactly: {MARK}: and one sentence on what is missing. "
    "(6) Be concise: at most 170 words, plain sentences, no headings.")


def format_evidence(items: list[EvidenceItem], max_chars: int) -> str:
    out, used = [], 0
    for it in items:
        loc = f", {it.location}" if it.location else ""
        block = f"[{it.id}] ({it.title}{loc}) {it.text.strip()[:760]}"
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)


def format_comparison(cmp_results: list[dict]) -> str:
    lines = []
    for r in cmp_results:
        for asp, pts in r.get("aspects", {}).items():
            for p in pts[:3]:
                lines.append(f"- {asp}: {p['sentence'][:260]} [{p['source_id']}]")
        for a in r.get("missing_aspects", []):
            lines.append(f"- NOT FOUND in sources: {a}")
        for c in r.get("conflicts", []):
            lines.append(f"- CONFLICT ({c.get('reason')}): [{c['a']['source_id']}] \"{c['a']['sentence'][:150]}\" vs [{c['b']['source_id']}] \"{c['b']['sentence'][:150]}\"")
    return "\n".join(lines)


def build_user_prompt(question: str, docs: list[EvidenceItem], computed: list[EvidenceItem], comparison: list[dict], gaps: list[str]) -> str:
    parts = []
    max_chars = cfg("agent")["budgets"]["max_evidence_chars_in_prompt"]
    if docs:
        parts.append("Document evidence:\n" + wrap_untrusted(format_evidence(docs, max_chars)))
    if computed:
        parts.append("Computed results (deterministic tool outputs, exact):\n" + "\n".join(f"[{c.id}] {c.text}" for c in computed))
    cmp = format_comparison(comparison)
    if cmp:
        parts.append("Comparison findings (extracted from the evidence above):\n" + wrap_untrusted(cmp))
    if gaps:
        parts.append("Parts of the question with NO supporting evidence found:\n" + "\n".join(f"- {g}" for g in gaps))
    return f"Question: {question}\n\n" + "\n\n".join(parts) + "\n\nAnswer:"


def _clean(raw: str, evidence: dict) -> str:
    """Remove citation ids that do not exist (never allow invented citations); normalise citation placement."""
    raw = re.sub(r"([.!?])\s*((?:\[[SDCG]\d+(?:\s*,\s*[SDCG]\d+)*\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", raw)
    def fix(m):
        ids = [x.strip() for x in re.split(r"[,;]", m.group(1))]
        good = [i for i in ids if i in evidence]
        return f"[{','.join(good)}]" if good else ""
    out = re.sub(r"\s*" + CITE.pattern, lambda m: " " + fix(m) if fix(m) else "", raw)
    out = re.sub(r"(\[[SDCG]\d+(?:,[SDCG]\d+)*\])(?:\s*\1)+", r"\1", out)
    return re.sub(r"[ \t]{2,}", " ", out).replace(" .", ".").strip()


def attribute_uncited(text: str, evidence: dict, candidates: list[str]) -> tuple[str, int, int]:
    """Deterministic citation repair: an uncited factual sentence gets the id of the candidate evidence item
    that demonstrably supports it (same support test as the validator); if none does, the sentence is dropped.
    Returns (text, n_attributed, n_dropped)."""
    out, att, drop = [], 0, 0
    for raw in SENT.split(text):
        sent = raw.strip()
        if not sent:
            continue
        if ids_in(sent) or len(strip_cites(sent)) < 25 or sent.startswith(("**", "Computed")):
            out.append(sent)
            continue
        best = None
        for cid in candidates:
            ok, sim, _ = sentence_supported(sent, [evidence[cid].text])
            if ok and (best is None or sim > best[1]):
                best = (cid, sim)
        if best:
            out.append(sent.rstrip() + f" [{best[0]}]")
            att += 1
        else:
            drop += 1
    return " ".join(out), att, drop


def synthesize(llm, question: str, docs: list[EvidenceItem], computed: list[EvidenceItem], comparison: list[dict], gaps: list[str],
               evidence: dict, max_regen: int = 1) -> dict:
    """Returns dict(answer, abstained, report(ValidationReport), raw, removed, regenerated, injection_flags)."""
    user = build_user_prompt(question, docs, computed, comparison, gaps)
    flags = sorted({f for d in docs for f in injection_flags(d.text)})
    raw = llm.chat(SYSTEM, user)
    regenerated = 0
    qn = norm_numbers(question)
    for _ in range(3):
        if raw.upper().startswith(MARK):
            return {"answer": raw.split(":", 1)[-1].strip() or "The knowledge base does not contain this information.", "abstained": True,
                    "report": ValidationReport(), "raw": raw, "removed": [], "regenerated": regenerated, "injection_flags": flags}
        cleaned = _clean(raw, evidence)
        rep = validate_answer(cleaned, evidence, qn)
        if regenerated < max_regen and (rep.unsupported() and rep.support_rate < 0.7):
            bad = "; ".join(f"\"{c.sentence[:90]}\" ({c.reason})" for c in rep.unsupported()[:3])
            raw = llm.chat(SYSTEM, user + f"\n\nYour previous draft had unsupported statements: {bad}. Rewrite using only statements that are directly stated in the evidence.")
            regenerated += 1
            continue
        break
    cands = [d.id for d in docs] + [c.id for c in computed]
    cleaned, n_att, n_drop = attribute_uncited(cleaned, evidence, cands)
    rep = validate_answer(cleaned, evidence, qn)
    # final guarantee: drop sentences whose citations do not support them
    removed = []
    if rep.unsupported():
        bad = {c.sentence for c in rep.unsupported()}
        kept = []
        for s in SENT.split(cleaned):
            if s.strip(" -*•\t") in bad:
                removed.append(s.strip())
            else:
                kept.append(s)
        cleaned = " ".join(kept).strip()
        rep = validate_answer(cleaned, evidence, qn)
    fallback = False
    if not cleaned.strip() and (docs or computed):
        # the LLM draft was entirely rejected, but sufficiently relevant evidence exists: answer extractively (verbatim evidence + ids)
        import numpy as np
        from src.embeddings.embedding_service import embed
        rows = [(d.id, x.strip()) for d in docs for x in SENT.split(d.text) if 30 < len(x.strip()) < 400]
        if rows:
            E = embed([question] + [r[1] for r in rows])
            top = np.argsort(-(E[1:] @ E[0]))[:2]
            cleaned = " ".join(f"{rows[i][1]} [{rows[i][0]}]" for i in sorted(top))
            fallback = True
            rep = validate_answer(cleaned, evidence, qn)
    return {"answer": cleaned, "abstained": False, "report": rep, "raw": raw, "fallback_extractive": fallback, "removed": removed, "regenerated": regenerated,
            "injection_flags": flags, "attributed": n_att, "dropped_uncited": n_drop}


def abstention_message(searched: list[str], best: str | None, gaps: list[str]) -> str:
    m = "I could not find sufficiently supported evidence in the selected knowledge base to answer this question."
    if gaps:
        m += " Missing: " + "; ".join(gaps[:3]) + "."
    if searched:
        m += " Searched for: " + "; ".join(f"'{q}'" for q in searched[:3]) + "."
    if best:
        m += f" Closest material found: {best} (not relevant enough to answer)."
    return m
