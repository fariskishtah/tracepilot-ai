# Knowledge base

35 documents, 4 courses, 4 file types, 167 chunks (all-MiniLM-L6-v2, 384-d, cosine).

| Course | Documents |
|---|---|
| `python` | lecture notes (PDF), exercise guide + FAQ/style addendum (DOCX), reference notes + glossary (TXT), debugging playbook (PDF), practice exercises + library cheatsheet (CSV) |
| `statistics` | lecture notes, estimators & bias (PDF), exercise guide, lab report guide (DOCX), reference notes, confusable terms, **robust summaries** (TXT), exam scores, worked examples (CSV) |
| `ml` | lecture notes, diagnostics handbook, **overfitting mitigation guide** (PDF), exercise guide, project handbook (DOCX), reference notes, notation, evaluation protocol v1 *(superseded)* (TXT), evaluation runs, hyperparameters, **regularization experiments** (CSV) |
| `data` | wrangling handbook (PDF), lab guide (DOCX), glossary, 2021 lab notes *(superseded)* (TXT), orders, pandas defaults, **product prices** (CSV) |

* 31 files are original CC BY 4.0 material written by the project owner for CourseAtlas AI / ReGround AI and copied here (attribution kept in `data/manifests/sources.json`).
* 4 files (bold) were written for TracePilot (`scripts/prepare_data.py`): they create genuine multi-step needs (mean ≠ median on skewed prices; mitigation techniques + experiment table).
* Deliberate traps: superseded documents that conflict with current ones (protocol v1/v2, 2021 vs 2025 lab rules, PEP 8 vs course line length), overloaded terms (variance, sample, precision, mode, index) and three **prompt-injection lines** inside course files.
* CSV chunks are text rows (`row 3: run_id=R03; …`) with real row ranges, and the registry stores each CSV's columns so the planner can pick datasets without a tool call.

Ingestion: extension + size (5 MB) + magic-byte checks, symlinks and macro formats rejected, text extracted only (nothing is executed), header/footer stripping, de-hyphenation, sentence-aware chunks (900 chars, 120 overlap),
SHA-256 duplicate detection (also across file names), update-in-place, delete, reindex, per-chunk content hash, safe storage names, course isolation via metadata filters.
