# Changes after the evaluation freeze (commit eb6d241)

The headline results were produced by the frozen code. Afterwards:

1. `src/agents/controller.py::_evaluate` — the retry of a **timed-out** tool call is now checked before the generic `analysis_error` repair. Found by the real-model integration test
   `test_K_tool_timeout_is_recoverable`. Impact on the reported benchmark: **none** — 0 tool timeouts occurred in all 288 recorded traces of the test runs (verified), so no evaluated behaviour changes.
2. UI-only fixes and documentation (if any) are listed in the git history after the freeze commit.

Bugs that *would* change test behaviour (dropped dataset filter, correlation argument repair, duplicate-id plan crash) were deliberately **not** fixed after the freeze; they are documented in docs/FAILURE_ANALYSIS.md.

## Verified file-level diff since the freeze (`git diff --name-only eb6d241 -- src configs data/evaluation/questions.jsonl data/documents scripts/evaluate.py`)

- `src/agents/controller.py`
- `src/ui/app_ui.py`
- `src/ui/theme.py`

Everything else under `src/` (planner, routing, replanning, synthesis, tools, citations, retrieval, evaluation) and all of `configs/`, the benchmark questions and the corpus are byte-identical to the frozen commit.
