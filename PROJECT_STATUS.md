# PROJECT STATUS — TracePilot AI

*Last updated: 2026-11-21*

| area | status |
|---|---|
| Academic assignment (Sprint 4, Project 4 — Agentic RAG) | **COMPLETE** — all 19 official requirements implemented and exercised |
| Engineering | **COMPLETE** (103 unit + 16 real-model integration + browser verification, all passing) |
| Public release | **PARTIAL** — GitHub published; Hugging Face Space **blocked** (HTTP 402) |
| Social media | LinkedIn: **PENDING** · Facebook: **PENDING** (handled separately) |

## Verified
* Knowledge base: 35 documents, 4 courses, 4 formats, 167 chunks, MiniLM embeddings, persistent ChromaDB (restart tested).
* Agent: LLM planner + guard, state machine, bounded replanning, abstention; 7 read-only tools; structured traces.
* Studio: 9 modules, verified in a real browser (15/15 checks, 12 screenshots).
* Evaluation: 114-question benchmark (frozen before test runs), Standard RAG vs agent + 3 ablations on the 72-question held-out split, paired bootstrap, failure analysis.

## Measured result (held-out test, rubric-based, AI-assisted annotation)
See README / `docs/EVALUATION.md`: correctness 42/72 (58.3%) → 53/72 (73.6%), +15.3 pp (95% CI +4.2…+26.4); numeric accuracy 25% → 78.6%; **not** better on comparison questions or answer-level multi-step completion; replanning gain small and unconfirmed.

## Known limitations
3B-model planning errors; benchmark annotation not human-audited; citation check is not NLI; ambiguous questions not detected; multi-step latency; no hosted demo; Qwen2.5-3B research licence.

## Post-freeze changes
See `data/evaluation/POST_FREEZE_CHANGES.md` (timeout-retry ordering fix, UI CSS fixes). Test results were produced by commit `eb6d241`.
