# Agent design

## 1. Question analyzer
`routing.analyze_question` extracts cheap rule features (calculation cue, dataset cue, comparison cue, summarization cue, multi-part cue,
named datasets, numbers). They are traced (`question_analysis`) and used by the guard; they do **not** pick the plan.

## 2. Planner
The LLM (`qwen2.5:3b-instruct` via Ollama) receives the enabled tools, the dataset catalog (ids, columns, row counts), the selected courses and three
few-shot plans (topics deliberately absent from the evaluation set), and returns an `LLMPlan` (JSON schema → Pydantic):
`question_type`, `complexity`, `requires_*`, and ≤ 8 `subtasks` `{id, tool, goal, query | dataset_id/operation/column/filters | expression, depends_on, aspects}`.
Different questions get different plans (see `tests/test_planning.py`, `docs/EVALUATION.md`).

## 3. Plan guard (hard constraints)
LLM plans are repaired, never trusted: unknown/disabled tools removed; dependencies must precede dependents; compare/summarize may only depend on
retrieval steps; analysis steps that cannot run (bad operation, missing column/group_by) are removed; invalid calculator variables are rewritten from
the available numeric steps; explanatory clauses ("explain why…", "retrieve …") that the plan did not cover get a search step; a comparison question with
≥2 searches gets a compare step; plans are truncated to 7 steps; an empty plan becomes one search. Every adjustment is written to the trace.
The same module also contains the **rule planner** (used as fallback and as an ablation).

## 4. Execution controller (`controller.py`)
States: `ANALYZE → PLAN → EXECUTE → VALIDATE → (REPLAN → EXECUTE)* → SYNTHESIZE → CITATION_CHECK → DONE`.
Per step it resolves arguments from earlier results (e.g. calculator variables `t1`, `t2` → evidence ids `D1`, `D2`; compare sources → best hits of the search
dependencies), calls the registry, stores the observation and evidence, and records `tool_selected`/`tool_result` events.

**Budgets** (`configs/agent.yaml`): ≤ 9 tool calls, ≤ 2 replanning rounds (initial plan + 2 replans), ≤ 3 attempts per subtask, ≤ 7 plan steps, 30 s per tool,
180 s per request. Exhaustion produces an explicit `stop_reason` (`tool_budget_exhausted`, `timeout`, …) and is traced.
These limits were set from the shape of the plans (a 4-part question needs 4–6 steps plus retries) and the 8 GB/latency budget; they were **not** tuned on the test split.

## 5. Validation and replanning
After each execution pass, `_evaluate` checks:
* search steps: dense cosine of the best hit ≥ 0.35 **and** keyword coverage of the top-3 hits ≥ 0.35 (thresholds fixed from the development split; see EVALUATION.md);
* failed `analyze_dataset` steps and timeouts (recoverable);
* a question that needs dataset numbers but has no analysis step.

Replan actions are deterministic and traced: `re_search` with a *different* query (ladder: synonym expansion → explanatory rephrase → most specific keywords),
`repair_arguments` (column fuzzy-matching, row selection from the step goal, filter-column repair), `retry` on timeout, `add_steps` (rule-based analysis plan).
A search that is still weak after its attempts is *not* used as evidence: that part of the question is passed to the synthesizer as a gap.
If nothing usable remains the agent **abstains** with a deterministic message listing what was searched.

## 6. Synthesis and citation check
One LLM call receives only sufficiently supported evidence (wrapped as untrusted data), exact computed results and extractive comparison findings, and must cite ids.
Post-processing (`synthesis.py`, `citations/validator.py`):
1. cite ids that do not exist are removed; citations after a period are normalised;
2. each cited sentence is checked: numbers must occur in the cited evidence (rounding-aware) **and** cosine ≥ 0.6 to a window of the evidence **and** content-word recall ≥ 0.5;
3. one regeneration if the support rate < 0.7; remaining unsupported sentences are dropped (the raw/removed counts are traced);
4. uncited factual sentences are attributed to the evidence item that demonstrably supports them, otherwise dropped;
5. computed results are always appended deterministically with their `[D#]/[C#]` ids, so numbers never depend on the LLM copying them.

Limitation: the support test is an embedding + lexical proxy, **not** an NLI model. It rejects off-topic and invented claims but can accept a contradiction that reuses the evidence's vocabulary.

## 7. What is *not* in the trace
Only action-level records are stored (plan, tool name/arguments (redacted, truncated), status, timing, evidence ids, replan actions, citation-check counts). Raw model reasoning is never stored or displayed.
