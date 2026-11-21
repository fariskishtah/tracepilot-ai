"""Agent state: typed plan, step records, budgets and the structured execution trace.
The trace stores action-level summaries and tool records only - never raw model reasoning."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.tools.base import EvidenceStore


class SubtaskFilter(BaseModel):
    column: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "contains"] = "=="
    value: str | float | int


class Subtask(BaseModel):
    """One planned action. Flat optional fields (easy for small models); validated again per tool."""
    id: str = Field(pattern=r"^t\d{1,2}$")
    tool: str
    goal: str = Field(max_length=240)
    query: str | None = Field(default=None, max_length=300)
    dataset_id: str | None = None
    operation: str | None = None
    column: str | None = None
    group_by: str | None = None
    agg: str | None = None
    filters: list[SubtaskFilter] | None = None
    expression: str | None = Field(default=None, max_length=200)
    aspects: list[str] | None = None
    depends_on: list[str] = Field(default_factory=list)


class QuestionPlan(BaseModel):
    question_type: Literal["factual", "comparison", "calculation", "data_analysis", "multi_step", "summarization", "unanswerable"]
    complexity: Literal["simple", "moderate", "complex"]
    requires_retrieval: bool
    requires_calculation: bool
    requires_data_analysis: bool
    requires_comparison: bool
    subtasks: list[Subtask] = Field(min_length=1, max_length=8)
    rationale: str = Field(default="", max_length=240)

    @field_validator("subtasks")
    @classmethod
    def unique_ids(cls, v):
        ids = [s.id for s in v]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate subtask ids")
        return v


class LLMPlan(BaseModel):
    """Schema handed to the LLM: no free-text rationale (small models ramble; the controller writes summaries)."""
    question_type: Literal["factual", "comparison", "calculation", "data_analysis", "multi_step", "summarization", "unanswerable"]
    complexity: Literal["simple", "moderate", "complex"]
    requires_retrieval: bool
    requires_calculation: bool
    requires_data_analysis: bool
    requires_comparison: bool
    subtasks: list[Subtask] = Field(min_length=1, max_length=8)


@dataclass
class StepRecord:
    subtask: Subtask
    status: str = "pending"          # pending | running | done | failed | skipped | weak
    attempts: int = 0
    result: Any = None               # ToolResult of the latest attempt
    queries_used: list[str] = field(default_factory=list)
    sufficient: bool | None = None
    note: str = ""


@dataclass
class Budgets:
    max_replans: int
    max_tool_calls: int
    max_llm_calls: int
    max_attempts_per_subtask: int
    request_timeout_s: float
    max_plan_steps: int
    tool_calls: int = 0
    replans: int = 0


class Trace:
    def __init__(self, question: str, scope: list[str] | None):
        self.request_id = uuid.uuid4().hex[:12]
        self.t0 = time.perf_counter()
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.events: list[dict] = []
        self.question, self.scope = question, scope

    def add(self, type_: str, **data) -> dict:
        ev = {"seq": len(self.events) + 1, "t_ms": round((time.perf_counter() - self.t0) * 1000), "type": type_, **data}
        self.events.append(ev)
        return ev


@dataclass
class AgentState:
    question: str
    scope: list[str] | None
    evidence: EvidenceStore
    trace: Trace
    budgets: Budgets
    plan: QuestionPlan | None = None
    plan_source: str = ""
    steps: dict[str, StepRecord] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    stop_reason: str = ""
    errors: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    def elapsed(self) -> float:
        return time.perf_counter() - self.trace.t0


@dataclass
class AgentResult:
    request_id: str
    question: str
    answer: str
    abstained: bool
    stop_reason: str
    plan: dict | None
    plan_source: str
    tool_sequence: list[str]
    citations: list[dict]
    evidence: dict
    trace: list[dict]
    validation: dict
    metrics: dict
    system: str = "agentic"
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)
