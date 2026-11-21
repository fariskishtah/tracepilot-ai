# Licences and provenance

| Component | Licence | Notes |
|---|---|---|
| TracePilot AI source code | MIT (`LICENSE`), © 2026 Faris Mohamed Kishtah | |
| `data/documents` (35 files) | CC BY 4.0 | 31 files: original material by the project owner (CourseAtlas AI / ReGround AI); 4 files written for TracePilot. `data/manifests/sources.json` lists origin, SHA-256 and attribution per file. Not official university material. |
| `data/evaluation` | CC BY 4.0 | AI-assisted annotation (see EVALUATION.md) |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 (verified on the Hub model card, revision `1110a243…`) | downloaded at first run, not redistributed |
| `qwen2.5:3b-instruct` (Ollama build of Qwen2.5-3B-Instruct) | **Qwen Research License** (`license: other / qwen-research`, verified on the Hub model card) | non-commercial research/education use; weights are **not** redistributed. The 3B model is a *different licence* from the 1.5B model. |
| `qwen2.5:1.5b-instruct` | Apache-2.0 (verified) | licence-clean alternative: `TRACEPILOT_LLM_MODEL=qwen2.5:1.5b-instruct`. Not used for the reported benchmark unless stated. |
| Ollama | MIT | local runtime |
| ChromaDB, Gradio, pandas, NumPy, Pydantic, rank-bm25, python-docx, sentence-transformers, Playwright, pytest, ReportLab | Apache-2.0 / BSD / MIT | |
| PyMuPDF | **AGPL-3.0** (or commercial) | unmodified dependency, not bundled. Offering a combined work as a network service triggers AGPL obligations – replace with `pypdf` for such use. |

No pretrained model was trained or fine-tuned in this project. Third-party downloads (models, browsers) are excluded from the repository and the submission archive.
