"""Application service: one KB / retriever / LLM shared by the UI, scripts and tests."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from src.agents.baseline import StandardRAG
from src.agents.controller import AgentConfig, TracePilotAgent
from src.config import ROOT, runtime_dir
from src.knowledge.indexing import KnowledgeBase
from src.knowledge.retrieval import Retriever
from src.llm import OllamaLLM


class Service:
    def __init__(self, auto_index: bool = True):
        self.kb = KnowledgeBase()
        self.retriever = Retriever(self.kb)
        self.llm = OllamaLLM()
        self._agents: dict[str, TracePilotAgent] = {}
        self.baseline = StandardRAG(self.kb, self.retriever, self.llm)
        self.lock = threading.Lock()          # one LLM job at a time on an 8 GB machine
        if auto_index and not self.kb.list_documents():
            self.kb.index_directory()

    def agent(self, name: str = "full", **kw) -> TracePilotAgent:
        key = name + json.dumps(kw, sort_keys=True, default=str)
        if key not in self._agents:
            self._agents[key] = TracePilotAgent(self.kb, self.retriever, self.llm, AgentConfig(name=name, **kw))
        return self._agents[key]

    def recent_traces(self, n: int = 200) -> list[dict]:
        d = runtime_dir() / "traces"
        out = []
        for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:n]:
            try:
                out.append(json.loads(f.read_text()))
            except (OSError, json.JSONDecodeError):
                pass
        return out


def load_metrics(name: str) -> dict | None:
    p = ROOT / "artifacts" / "metrics" / name
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
