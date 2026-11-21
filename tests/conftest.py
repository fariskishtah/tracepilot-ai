import json
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["ml/ml_lecture_notes.pdf", "ml/ml_overfitting_mitigation_guide.pdf", "ml/ml_regularization_experiments.csv", "ml/ml_project_handbook.docx",
         "ml/ml_evaluation_protocol_v1.txt", "ml/ml_reference_notes.txt", "data/product_prices.csv", "data/data_orders.csv", "python/python_reference_notes.txt",
         "statistics/statistics_lecture_notes.pdf", "statistics/statistics_robust_summaries.txt"]


class FakeLLM:
    """Deterministic stand-in so state-machine logic can be unit-tested without Ollama."""
    model = "fake"

    def __init__(self, plan: dict | None = None, answer: str | None = None, fail_plan: bool = False):
        from src.llm import LLMUsage
        self.usage, self.plan, self.answer, self.fail_plan = LLMUsage(), plan, answer, fail_plan
        self.prompts: list[str] = []

    def structured(self, system, user, schema_model, max_tokens=None, retry_seed=False):
        from src.llm import LLMError
        self.usage.calls += 1
        self.prompts.append(user)
        if self.fail_plan or self.plan is None:
            raise LLMError("fake planner failure")
        return schema_model.model_validate(self.plan)

    def chat(self, system, user, max_tokens=None):
        self.usage.calls += 1
        self.prompts.append(user)
        if self.answer is not None:
            return self.answer
        m = re.search(r"\[(S\d+)\] \([^)]*\) ([^\n]{50,160})", user)
        if m:
            return m.group(2).rstrip(" .,;") + f" [{m.group(1)}]."
        m = re.search(r"\[([DC]\d+)\] ([^\n]{10,120})", user)
        return f"{m.group(2)} [{m.group(1)}]." if m else "INSUFFICIENT_EVIDENCE: nothing to answer with."


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    for rel in SMALL:
        (d / Path(rel).parent).mkdir(exist_ok=True)
        shutil.copy(ROOT / "data" / "documents" / rel, d / rel)
    return d


@pytest.fixture(scope="session")
def kb(tmp_path_factory, corpus_dir):
    from src.knowledge.indexing import KnowledgeBase
    from src.knowledge.vector_store import VectorStore
    rt = tmp_path_factory.mktemp("rt")
    k = KnowledgeBase(VectorStore(rt / "chroma"), rt / "registry.sqlite", rt / "uploads")
    k.index_directory(corpus_dir)
    return k


@pytest.fixture(scope="session")
def retriever(kb):
    from src.knowledge.retrieval import Retriever
    return Retriever(kb)


@pytest.fixture()
def ctx(kb, retriever):
    from src.tools.base import EvidenceStore, ToolContext
    return ToolContext(kb, retriever, FakeLLM(), EvidenceStore())


@pytest.fixture(scope="session")
def registry():
    from src.tools.registry import build_registry
    return build_registry()


def make_agent(kb, retriever, llm, **cfg):
    from src.agents.controller import AgentConfig, TracePilotAgent
    import tempfile
    return TracePilotAgent(kb, retriever, llm, AgentConfig(**cfg), trace_dir=Path(tempfile.mkdtemp()))
