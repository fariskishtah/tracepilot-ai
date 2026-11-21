from __future__ import annotations

from src.config import cfg
from src.tools.base import ToolRegistry
from src.tools.calculator import CalculateExpression
from src.tools.comparison import CompareEvidence
from src.tools.data_analysis import AnalyzeDataset
from src.tools.knowledge_search import SearchKnowledgeBase
from src.tools.query_refinement import RefineSearchQuery
from src.tools.source_retrieval import RetrieveSourceContext
from src.tools.summarization import SummarizeEvidence


def build_registry(enabled: list[str] | None = None) -> ToolRegistry:
    reg = ToolRegistry()
    for t in (SearchKnowledgeBase(), RetrieveSourceContext(), SummarizeEvidence(), AnalyzeDataset(),
              CalculateExpression(), CompareEvidence(), RefineSearchQuery()):
        reg.register(t)
    reg.set_enabled(enabled if enabled is not None else cfg("agent")["tools_enabled"])
    return reg
