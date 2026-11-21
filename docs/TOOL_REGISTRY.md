# Tool registry (version 1.0.0)

All tools are **read-only**, typed with Pydantic, run in a worker thread with a timeout (30 s) and a bounded output (6000 chars). The agent can only call tools that are
registered **and** enabled; there is no shell, file, network or SQL tool. Unknown fields in arguments are ignored; invalid ones return `invalid_arguments`.

| Tool | Purpose | Key inputs | Output / provenance |
|---|---|---|---|
| `search_knowledge_base` | dense (Chroma) + BM25 fused retrieval | `query`, `course_ids`, `doc_ids`, `file_types`, `top_k ≤ 8` | hits with `source_id` (S#), doc, course, page/section/row range, cosine. The user's course scope can be narrowed but never widened by the tool call. |
| `retrieve_source_context` | neighbouring chunks of a retrieved source | `source_id`, `window ≤ 2` | chunks of the **same** document; refuses sources outside the course scope |
| `summarize_evidence` | summarize chosen sources | `source_ids`, `focus` | bullets with ids; each bullet must pass the support check or an extractive fallback is used; result is a `G#` *generated* item that is never primary evidence |
| `analyze_dataset` | deterministic CSV statistics | `dataset_id`, `operation` ∈ {row_count, describe_columns, missing_values, mean, median, min, max, sum, std, var, quantile, count, value_counts, groupby_agg, correlation, select_rows, lookup_value}, `column`, `group_by`, `agg`, `filters ≤ 5`, `ddof` | value/table + file, SHA-256 prefix, rows used/total, filters; `D#`. `std/var` are **sample** statistics (ddof=1) unless `ddof=0`. |
| `calculate_expression` | safe arithmetic | `expression ≤ 200 chars`, `variables` (numbers or evidence ids) | value + inputs + input evidence ids; `C#`. AST allowlist; no `eval`; exponent/size/finite checks |
| `compare_evidence` | compare retrieved sources | `source_ids (2–8)`, `aspects ≤ 4` | per-aspect verbatim sentences, agreements, conflicts (numeric differences or superseded/outdated markers), missing aspects. Fully extractive. |
| `refine_search_query` | decompose / normalise / expand queries | `question`, `previous_queries`, `mode` | ≤ 4 queries; a rewrite is rejected if numbers, identifiers or negations are lost or numbers are invented |

Evidence ids: `S#` document passage · `D#` dataset computation · `C#` calculation · `G#` generated summary. Only ids minted by tool outputs can be cited.

Errors are returned as `ToolResult(ok=False, error_type ∈ invalid_arguments | tool_error | timeout | disabled | unknown_tool)` with a short safe message (no stack traces or paths).
