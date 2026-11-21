# Execution workflow (worked example, real trace)

Question: *Find the definition of overfitting, retrieve two mitigation techniques, compare them, and calculate the difference between the validation accuracies of run R02 and run R04 in the regularization experiments dataset.*

Typical recorded sequence (see `artifacts/execution_traces/` for complete JSON traces):

1. `request_received` – scope, enabled tools, model, registry version.
2. `question_analysis` – calc/data/compare/multi-part cues.
3. `plan_created` – `source: llm`, subtasks, plus any guard `adjustments`.
4. `tool_selected` / `tool_result` per step: `search_knowledge_base` (definition), `search_knowledge_base` (mitigation), `compare_evidence`, `analyze_dataset` ×2 (`lookup_value` of `val_accuracy` for R02 and R04), `calculate_expression` (`t1 - t2`).
5. `validation` – per-step status and sufficiency.
6. (`replan` when a step is weak/failed – re-search with a new query, argument repair, retry.)
7. `synthesis_started` → `citation_check` → `answer` → `completed` (stop reason, latency).

The Studio shows the same events as a live timeline, an executed-graph SVG, an evidence inspector and a downloadable JSON trace.
