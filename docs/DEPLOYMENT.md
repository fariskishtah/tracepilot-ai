# Deployment

## Local (the supported deployment)

```bash
ollama serve &                      # local only; never expose it
ollama pull qwen2.5:3b-instruct
python scripts/build_index.py
python app.py                       # Gradio on 127.0.0.1:7860 (override with TRACEPILOT_PORT)
```

Memory on an 8 GB M2: the 3B model (~2 GB) + MiniLM + ChromaDB fit comfortably; the app runs one LLM job at a time (`Service.lock`). Use `TRACEPILOT_LLM_MODEL=qwen2.5:1.5b-instruct` on smaller machines (Apache-2.0, lower accuracy — not benchmarked here).

## Hugging Face Spaces — checked, **not available for this account**

* `fariskishtah/tracepilot-ai` does not exist as a Space (checked with the Hub API, 2026-11-21).
* A live attempt to create a **private Gradio Space** (`space_sdk="gradio"`, free CPU) returned **HTTP 402 Payment Required** ("hosting Gradio and Docker Spaces … requires a PRO subscription"); only *static* Spaces are free. `whoami` reports a non-PRO user account. The same 402 was recorded for ReGround AI.
* Nothing was created, no subscription was bought and the account was not upgraded.
* A static Space cannot run retrieval, tools or an LLM, so publishing one would **not** be a live Agentic RAG application and was not done.
* Even with a paid Space, the agent needs a local LLM runtime: free CPU hardware could not run Ollama, and a hosted-LLM API would violate the "no paid inference" constraint.

Therefore the public artefacts are the GitHub repository (code, data, tests, results), authentic screenshots, a PDF report and full reproduction steps. Ollama is never exposed publicly.
