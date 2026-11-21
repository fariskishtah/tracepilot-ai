# Architecture

TracePilot AI is a local, bounded **Agentic RAG** system. Everything runs on one machine (Apple Silicon, 8 GB): Ollama for the LLM,
sentence-transformers for embeddings, ChromaDB for vectors, Gradio for the UI. No paid service is used.

## RAG generations (and what is new here)

| | Standard RAG (CourseAtlas AI) | Corrective RAG (ReGround AI) | **Agentic RAG (TracePilot AI)** |
|---|---|---|---|
| Control flow | fixed: retrieve → generate | fixed + one correction loop | **planned per question**, conditional |
| Retrieval | one query | grade → rewrite → re-retrieve | decomposed sub-queries, retries only when evidence is weak |
| Tools | none | none | search, source context, summarize, **dataset analysis, calculator**, compare, refine |
| Numbers | read from text | read from text | **computed deterministically** (pandas / safe calculator) |
| Multi-step | no | no | dependency graph of steps, evidence aggregation |
| Failure handling | abstain | re-retrieve | targeted repair, bounded replanning, abstention |

ReGround AI's own evaluation did **not** statistically confirm an accuracy gain over Standard RAG; no such claim is inherited here.
TracePilot is a separate code base. CourseAtlas/ReGround directories were only read; the documents were *copied* (CC BY 4.0, own work).

## Components

```
question ─► QUESTION ANALYZER ─► PLANNER (LLM, JSON-schema) ─► PLAN GUARD ─► EXECUTION CONTROLLER
              (rule features)      typed QuestionPlan          hard rules     (state machine, budgets)
                                                                                   │
                         ┌──────────────┬───────────────┬───────────────┬──────────┴──────────┐
                  search_knowledge_base  analyze_dataset  calculate_expression  compare / summarize / context
                         └──────────────┴───────► EVIDENCE STORE (S#, D#, C#, G# ids) ◄──────────┘
                                                                  │
                                      VALIDATE (sufficiency) ─► REPLAN (≤2) ─► EXECUTE
                                                                  │
                                    SYNTHESIS (LLM, untrusted evidence) ─► CITATION CHECK ─► answer / abstention
```

* `src/ingestion` – validated loaders (PDF/DOCX/CSV/TXT), cleaning, sentence-aware chunking with provenance.
* `src/knowledge` – ChromaDB (cosine) persistent store, SQLite document registry (dedup/update/delete/reindex), hybrid retriever (dense + BM25 reciprocal-rank fusion) shared by the baseline and the agent.
* `src/tools` – typed read-only tools, `ToolRegistry` (Pydantic validation, allowlist, timeout, output cap), `EvidenceStore`.
* `src/agents` – `routing.py` (features, rule planner, plan guard), `planner.py` (LLM planner), `controller.py` (state machine), `replanning.py`, `synthesis.py`, `baseline.py` (Standard RAG).
* `src/citations` – citation validator (ids must exist **and** be supported).
* `src/security` – untrusted-content wrapping and injection detection.
* `src/ui` – TracePilot Studio; every panel is rendered from the structured trace/evidence of a real run.

## Why a custom state machine instead of LangGraph

LangGraph (v1.2.x on PyPI at the time of writing) offers persistence, streaming and interrupts. For this project those are not needed, while the costs are real:
it pulls in the LangChain dependency tree, abstracts exactly the layer we must instrument (every transition has to land in our own trace schema), and a
small explicit controller (~400 lines) makes the budgets, stop reasons and ablations trivial to test. The architecture is nevertheless graph-shaped
(plan steps form a dependency DAG) and could be ported to LangGraph without changing tools or evidence handling.

## Why schema-constrained JSON instead of native function calling

Ollama supports both structured outputs (JSON-schema `format`) and native tool calling. A *plan* is a multi-step object with dependencies, so the planner
uses schema-constrained generation, validated again with Pydantic; invalid output is **never** counted as a tool call. If the LLM output is invalid
twice (second attempt uses a different seed), the trace records `plan_source = rules_fallback` — the fallback is visible, not hidden.
