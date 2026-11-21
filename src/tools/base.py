"""Tool framework: typed tools, evidence store (verified provenance), registry with validation,
timeouts, size caps and allowlisting. Tools are read-only; there is no shell/file/network access."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Any, ClassVar, Type

from pydantic import BaseModel, ValidationError

from src.config import cfg


# ------------------------------------------------------------------ evidence
@dataclass
class EvidenceItem:
    id: str                    # S1 (document) | D1 (dataset op) | C1 (calculation) | G1 (generated summary)
    kind: str                  # document | dataset | calculation | summary
    title: str
    text: str                  # exact text that supports claims (chunk text / computed statement)
    doc_id: str | None = None
    course_id: str | None = None
    location: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    data: dict = field(default_factory=dict)
    derived_from: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class EvidenceStore:
    """Citations may only reference IDs minted here, from real tool outputs."""
    PREFIX = {"document": "S", "dataset": "D", "calculation": "C", "summary": "G"}

    def __init__(self) -> None:
        self.items: dict[str, EvidenceItem] = {}
        self._by_chunk: dict[str, str] = {}
        self._n = {"S": 0, "D": 0, "C": 0, "G": 0}

    def _mint(self, kind: str) -> str:
        p = self.PREFIX[kind]
        self._n[p] += 1
        return f"{p}{self._n[p]}"

    def add_hit(self, hit) -> EvidenceItem:
        if hit.chunk_id in self._by_chunk:
            it = self.items[self._by_chunk[hit.chunk_id]]
            it.score = max(it.score or 0, hit.score)
            return it
        m = hit.meta
        it = EvidenceItem(self._mint("document"), "document", m.get("title", m["doc_id"]), hit.text, m["doc_id"],
                          m.get("course_id"), hit.location(), hit.chunk_id, round(hit.score, 4),
                          {"file_type": m.get("file_type"), "page": m.get("page"), "section": m.get("section"),
                           "filename": m.get("filename")})
        self.items[it.id] = it
        self._by_chunk[hit.chunk_id] = it.id
        return it

    def add_computation(self, kind: str, title: str, text: str, data: dict, doc_id=None, course_id=None, location=None) -> EvidenceItem:
        it = EvidenceItem(self._mint(kind), kind, title, text, doc_id, course_id, location, data=data)
        self.items[it.id] = it
        return it

    def add_summary(self, text: str, derived_from: list[str], title="Generated summary") -> EvidenceItem:
        it = EvidenceItem(self._mint("summary"), "summary", title, text, derived_from=derived_from)
        self.items[it.id] = it
        return it

    def get(self, i: str) -> EvidenceItem | None:
        return self.items.get(i)

    def numeric(self, i: str) -> float | None:
        it = self.items.get(i)
        if it and it.kind in ("dataset", "calculation"):
            v = it.data.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return None

    def primary(self) -> list[EvidenceItem]:
        """Evidence that may support factual claims (generated summaries are excluded)."""
        return [i for i in self.items.values() if i.kind != "summary"]


# ------------------------------------------------------------------ context / results
@dataclass
class ToolContext:
    kb: Any
    retriever: Any
    llm: Any
    evidence: EvidenceStore
    allowed_courses: list[str] | None = None
    scratch: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: dict = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    error: str | None = None
    error_type: str | None = None          # invalid_arguments | tool_error | timeout | disabled | unknown_tool
    summary: str = ""
    duration_ms: float = 0.0
    args: dict = field(default_factory=dict)


class Tool:
    name: ClassVar[str]
    description: ClassVar[str]
    Input: ClassVar[Type[BaseModel]]
    output_description: ClassVar[str]
    examples: ClassVar[list[dict]] = []
    read_only: ClassVar[bool] = True
    uses_llm: ClassVar[bool] = False

    def run(self, args: BaseModel, ctx: ToolContext) -> tuple[dict, list[str], str]:
        """Returns (data, evidence_ids, one-line summary)."""
        raise NotImplementedError


def redact_args(args: dict, limit: int = 160) -> dict:
    """Arguments stored in traces: bounded strings, no raw document contents."""
    def r(v):
        if isinstance(v, str):
            return v if len(v) <= limit else v[:limit] + "…"
        if isinstance(v, dict):
            return {k: r(x) for k, x in v.items()}
        if isinstance(v, list):
            return [r(x) for x in v[:12]]
        return v
    return r(args)


class ToolRegistry:
    def __init__(self, enabled: list[str] | None = None):
        self._tools: dict[str, Tool] = {}
        self.enabled: set[str] = set(enabled or [])
        self.version = cfg("agent").get("registry_version", "1.0.0")
        self._pool = ThreadPoolExecutor(max_workers=2)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        if not self.enabled or True:
            self.enabled.add(tool.name)

    def set_enabled(self, names: list[str]) -> None:
        self.enabled = {n for n in names if n in self._tools}

    def names(self) -> list[str]:
        return [n for n in self._tools if n in self.enabled]

    def all_names(self) -> list[str]:
        return list(self._tools)

    def get(self, n: str) -> Tool | None:
        return self._tools.get(n)

    def specs(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.Input.model_json_schema(),
                 "output": t.output_description, "examples": t.examples, "read_only": t.read_only,
                 "enabled": t.name in self.enabled} for t in self._tools.values()]

    def validate(self, name: str, raw: dict) -> tuple[BaseModel | None, str | None]:
        t = self._tools.get(name)
        if t is None:
            return None, f"unknown tool '{name}'"
        try:
            return t.Input.model_validate(raw), None
        except ValidationError as e:
            return None, "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())[:400]

    def execute(self, name: str, raw: dict, ctx: ToolContext, timeout: float | None = None) -> ToolResult:
        a = cfg("agent")["budgets"]
        timeout = timeout or a["tool_timeout_s"]
        t0 = time.perf_counter()
        res = ToolResult(name, False, args=redact_args(raw if isinstance(raw, dict) else {}))
        tool = self._tools.get(name)
        if tool is None:
            res.error, res.error_type = f"unknown tool '{name}'", "unknown_tool"
        elif name not in self.enabled:
            res.error, res.error_type = f"tool '{name}' is disabled", "disabled"
        else:
            inp, err = self.validate(name, raw if isinstance(raw, dict) else {})
            if err:
                res.error, res.error_type = err, "invalid_arguments"
            else:
                fut = self._pool.submit(tool.run, inp, ctx)
                try:
                    data, ev, summ = fut.result(timeout=timeout)
                    res.ok, res.data, res.evidence_ids, res.summary = True, data, ev, summ
                except FutTimeout:
                    fut.cancel()
                    res.error, res.error_type = f"timeout after {timeout}s", "timeout"
                except ToolError as e:
                    res.error, res.error_type = str(e)[:300], "tool_error"
                except Exception as e:  # never leak internals/paths
                    res.error, res.error_type = f"{type(e).__name__}", "tool_error"
        res.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        if res.ok:
            res.data = _cap(res.data, a["max_tool_output_chars"])
        return res


class ToolError(RuntimeError):
    """Expected, recoverable tool failure with a safe message."""


def _cap(d: Any, limit: int) -> Any:
    """Bound output size: truncate long strings / lists so tool output cannot flood the context."""
    import json
    s = json.dumps(d, default=str)
    if len(s) <= limit:
        return d
    if isinstance(d, dict):
        return {k: _cap(v, max(200, limit // max(1, len(d)))) for k, v in d.items()}
    if isinstance(d, list):
        return [_cap(x, max(200, limit // max(1, len(d)))) for x in d[:20]]
    if isinstance(d, str):
        return d[:limit] + "…[truncated]"
    return d
