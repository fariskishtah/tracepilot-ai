# Standard RAG vs TracePilot (Agentic RAG)

Baseline A = Standard RAG: one hybrid retrieval over the question (top-5), one grounded generation with the **same** prompt, validator and abstention rule as the agent, no planning, no tools, no replanning.
System B = TracePilot (full): LLM planner + guard, tools, bounded replanning, deterministic computed results.

<!-- BEGIN:main -->
| metric (test split, n=72) | Standard RAG | TracePilot (agentic) |
|---|---|---|
| answer_correctness (all) | 42/72 (58.3%) | 53/72 (73.6%) |
| answer_correctness (answerable) | 36/66 (54.5%) | 47/66 (71.2%) |
| correct_abstention (unanswerable) | 6/6 (100.0%) | 6/6 (100.0%) |
| numerical_accuracy | 7/28 (25.0%) | 22/28 (78.6%) |
| multi_step_full_completion | 14/22 (63.6%) | 13/22 (59.1%) |
| mean latency (s) | 9.9 | 10.3 |
| p95 latency (s) | 29.9 | 24.7 |
| mean tool calls | 1.00 | 2.19 |
| false_abstention (answerable) | 8/66 (12.1%) | 6/66 (9.1%) |
| gold-document citation hit | 53/59 (89.8%) | 53/59 (89.8%) |
| mean LLM calls / tokens | 1.25 / 1296 | 2.25 / 2583 |

Paired difference in correctness: **+15.3 pp**, 95% paired-bootstrap CI **[+4.2, +26.4] pp** (TracePilot-only correct: 15, Standard-only correct: 4; interval excludes 0).
<!-- END:main -->

<!-- BEGIN:categories -->
| category | n | Standard RAG | TracePilot | paired diff (pp) |
|---|---|---|---|---|
| ambiguous | 2 | 1/2 | 1/2 | +0 |
| calculation | 5 | 1/5 | 4/5 | +60 |
| comparison | 9 | 8/9 | 5/9 | -33 |
| conflict | 2 | 1/2 | 1/2 | +0 |
| csv_analysis | 11 | 1/11 | 8/11 | +64 |
| injection | 4 | 4/4 | 3/4 | -25 |
| multi_step | 9 | 3/9 | 5/9 | +22 |
| simple_factual | 15 | 11/15 | 12/15 | +7 |
| summarization | 4 | 3/4 | 3/4 | +0 |
| tool_routing | 5 | 3/5 | 5/5 | +40 |
| unanswerable | 6 | 6/6 | 6/6 | +0 |
<!-- END:categories -->

## Where the agent helped
* **Anything that needs computation** — CSV analysis and calculation: Standard RAG can only quote numbers that happen to be written in text and cannot aggregate rows; the agent computes them with pandas / the calculator and attaches file hash, rows used and filters as provenance.
* **Tool-routing questions** and **correct abstention** (both systems abstained on all unanswerable test questions; the agent additionally explains what it searched).
* Cost of the agent: more LLM calls and tokens and more tool calls (table above).

## Where it did not help
* **Comparison questions**: Standard RAG did better in this run (see categories) — decomposition into per-side searches did not improve a 3B model's synthesis and the faithfulness filter shortened answers (FAILURE_ANALYSIS.md §5).
* **Multi-step full completion at answer level** was not better than Standard RAG: many multi-step items are text-retrievable, and agent runs lose points to wrong arguments and budget use.
* **Plain factual questions**: near parity, Standard RAG is cheaper and faster. Simple questions should (and do) use one search.

## Corrective RAG reference
ReGround AI (Corrective RAG) was not re-run here: its repository was deliberately left untouched and integrating it would not have changed the Standard-vs-Agentic conclusions. Its own result (accuracy gain over Standard RAG not statistically confirmed) is cited, not extended.

## Caveats
Same model and hardware for both; AI-assisted annotation; keyword rubric; per-category n ≤ 15; latency measured on a shared laptop. Differences in categories with n < 10 should be read as anecdotes, not as findings.
