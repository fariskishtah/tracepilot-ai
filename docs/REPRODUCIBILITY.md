# Reproducibility

Hardware used: Apple M2, 8 GB unified memory, macOS. Python 3.12.

```bash
git clone https://github.com/fariskishtah/tracepilot-ai && cd tracepilot-ai
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct            # exact LLM id used for all reported numbers
python scripts/prepare_data.py             # regenerates the 4 TracePilot documents + manifest (deterministic)
python scripts/build_index.py              # index data/documents into data/runtime (Chroma + SQLite)
python scripts/build_eval_set.py           # rebuilds data/evaluation (fails if any gold evidence pattern is missing)
python app.py                              # Studio at http://127.0.0.1:7860
python -m pytest -m "not integration and not browser"   # unit tests (no LLM)
python -m pytest -m integration            # real-model tests (needs Ollama)
python scripts/evaluate.py --system standard --split test
python scripts/evaluate.py --system full --split test
python scripts/summarize.py --split test --systems standard full
```

Determinism: LLM `temperature=0`, `seed=7`, fixed prompts; retrieval, tools, scoring and bootstrap (`seed=20261121`) are deterministic. Ollama/Metal inference is **not guaranteed bit-identical** across versions/hardware, so
re-running may change a few individual answers; the saved result files in `artifacts/metrics/` are the numbers reported in the docs.
Every result file has a `.meta.json` (model, embedding, configs, dataset version, corpus hash, code commit, date).
