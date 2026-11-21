# Ablation study (test split)

Research question: which agentic components contribute to correctness, at what latency cost?

| experiment | system id | description |
|---|---|---|
| A | `standard` | Standard RAG |
| B | `llm_no_numeric_tools` | LLM planner, **retrieval tools only** (search, source context, summarize, compare, refine) — no dataset analysis / calculator, no replanning |
| C | `rules_all_tools` | **rule-based planner** with all tools, no replanning |
| D = E | `llm_no_replan` | LLM planner + guard, all tools, **no replanning** |
| F | `full` | LLM planner + guard, all tools, bounded replanning (the reported system) |

All runs use the frozen code and configuration; only the `AgentConfig` differs (`scripts/evaluate.py` SYSTEMS). Because "full planning" (D) and "without replanning" (E) are the same configuration, they are reported once.

<!-- BEGIN:ablation -->
| system | n | correct | numeric | multi-step | mean latency (s) | tool calls |
|---|---|---|---|---|---|---|
| A  Standard RAG | 72 | 42/72 (58.3%) | 7/28 (25.0%) | 14/22 (63.6%) | 9.9 | 1.00 |
| B  LLM planner, retrieval tools only (no calc/data), no replan | 72 | 37/72 (51.4%) | 6/28 (21.4%) | 11/22 (50.0%) | 9.3 | 1.65 |
| C  rule planner, all tools, no replan | 72 | 38/72 (52.8%) | 12/28 (42.9%) | 9/22 (40.9%) | 4.7 | 1.53 |
| D/E  LLM planner, all tools, no replan | 72 | 51/72 (70.8%) | 20/28 (71.4%) | 12/22 (54.5%) | 9.0 | 1.92 |
| F  full: LLM planner + bounded replanning | 72 | 53/72 (73.6%) | 22/28 (78.6%) | 13/22 (59.1%) | 10.3 | 2.19 |

Paired bootstrap of (full − system) in correctness: vs standard: +15.3 pp, 95% CI [+4.2, +26.4]; vs llm_no_numeric_tools: +22.2 pp, 95% CI [+11.1, +33.3]; vs rules_all_tools: +20.8 pp, 95% CI [+11.1, +30.6]; vs llm_no_replan: +2.8 pp, 95% CI [+0.0, +6.9]. Experiments D and E coincide (an LLM planner without replanning); intervals that include zero mean the component's contribution is not confirmed on this benchmark.
<!-- END:ablation -->

Interpretation is in the README/report; intervals that include zero mean the component's contribution is **not confirmed** on this benchmark.
