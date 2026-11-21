"""Standard RAG baseline: ONE retrieval with the question, ONE generation. Same retriever, embedding model,
LLM, prompt, validator and post-processing as the agent - no planning, tools or replanning."""
from __future__ import annotations

import time

from src.agents.state import AgentResult, Trace
from src.agents.synthesis import abstention_message, synthesize
from src.citations.validator import ids_in
from src.config import cfg
from src.llm import LLMError, OllamaLLM
from src.tools.base import EvidenceStore


class StandardRAG:
    def __init__(self, kb, retriever, llm: OllamaLLM | None = None, top_k: int = 5):
        self.kb, self.retriever, self.llm, self.top_k = kb, retriever, llm or OllamaLLM(), top_k

    def run(self, question: str, course_ids: list[str] | None = None) -> AgentResult:
        tr = Trace(question, course_ids)
        ev = EvidenceStore()
        u0 = (self.llm.usage.calls, self.llm.usage.prompt_tokens, self.llm.usage.completion_tokens)
        tr.add("request_received", question=question, scope=course_ids, system="standard_rag")
        hits = self.retriever.search(question, self.top_k, course_ids)
        items = [ev.add_hit(h) for h in hits]
        tr.add("retrieval", query=question, n=len(items), sources=[i.id for i in items], best_score=round(max((h.score for h in hits), default=0), 3))
        stop, abstained, rep = "answered", False, {}
        try:
            r = synthesize(self.llm, question, items, [], [], [], ev.items)
            answer, abstained, rep = r["answer"], r["abstained"], r["report"].to_dict()
            if abstained:
                stop = "insufficient_evidence"
            elif not answer.strip():
                answer, abstained, stop = abstention_message([question], None, []), True, "insufficient_evidence"
        except LLMError as e:
            answer, abstained, stop = f"LLM unavailable ({e})", True, "llm_unavailable"
        cited = [i for i in dict.fromkeys(ids_in(answer)) if i in ev.items]
        tr.add("answer", abstained=abstained, citations=cited)
        lat = time.perf_counter() - tr.t0
        return AgentResult(tr.request_id, question, answer, abstained, stop, None, "none", ["search_knowledge_base"],
                           [ev.items[i].to_dict() for i in cited], {k: v.to_dict() for k, v in ev.items.items()}, tr.events, rep,
                           {"latency_s": round(lat, 2), "llm_calls": self.llm.usage.calls - u0[0], "prompt_tokens": self.llm.usage.prompt_tokens - u0[1],
                            "completion_tokens": self.llm.usage.completion_tokens - u0[2], "tool_calls": 1, "replans": 0, "steps": 1},
                           system="standard_rag", config={"top_k": self.top_k})
