"""Deterministic CSV analysis with an allowlist of operations. No eval/query strings/model-written code."""
from __future__ import annotations

import difflib
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.ingestion.loaders import sha256_bytes
from src.tools.base import Tool, ToolContext, ToolError
from src.tools.knowledge_search import scope_courses

Op = Literal["describe_columns", "row_count", "missing_values", "mean", "median", "min", "max", "sum", "std", "var",
             "count", "quantile", "value_counts", "groupby_agg", "correlation", "select_rows", "lookup_value"]
Agg = Literal["mean", "median", "sum", "min", "max", "count", "std"]


class Filter(BaseModel):
    column: str = Field(max_length=60)
    op: Literal["==", "!=", ">", ">=", "<", "<=", "contains"] = "=="
    value: str | float | int


class AnalyzeInput(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=80)
    operation: Op
    column: str | None = Field(default=None, max_length=60)
    other_column: str | None = Field(default=None, max_length=60)
    group_by: str | None = Field(default=None, max_length=60)
    agg: Agg | None = None
    filters: list[Filter] = Field(default_factory=list, max_length=5)
    ddof: Literal[0, 1] = 1                       # 1 = sample statistics (default), 0 = population
    q: float = Field(default=0.5, ge=0, le=1)
    limit: int = Field(default=10, ge=1, le=20)


NUMERIC_OPS = {"mean", "median", "min", "max", "sum", "std", "var", "quantile", "correlation"}


def _col(df: pd.DataFrame, name: str | None, what="column") -> str:
    if not name:
        raise ToolError(f"{what} is required for this operation")
    if name in df.columns:
        return name
    low = {c.lower(): c for c in df.columns}
    if name.lower() in low:
        return low[name.lower()]
    near = difflib.get_close_matches(name, list(df.columns), n=3, cutoff=0.5)
    raise ToolError(f"{what} '{name}' not found. Available columns: {list(df.columns)}" + (f" (did you mean {near}?)" if near else ""))


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    for f in filters:
        c = _col(df, f.column, "filter column")
        s = df[c]
        v: object = f.value
        if f.op == "contains":
            df = df[s.astype(str).str.contains(str(v), case=False, regex=False)]
            continue
        if pd.api.types.is_numeric_dtype(s):
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ToolError(f"filter value for numeric column '{c}' must be a number")
            m = {"==": s == v, "!=": s != v, ">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[f.op]
        else:
            if f.op not in ("==", "!="):
                raise ToolError(f"operator {f.op} needs a numeric column")
            ss, vv = s.astype(str).str.strip().str.lower(), str(v).strip().lower()
            m = (ss == vv) if f.op == "==" else (ss != vv)
        df = df[m]
    return df


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(df[c]):
        raise ToolError(f"column '{c}' is not numeric (dtype {df[c].dtype})")
    return df[c].dropna()


def _py(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    return x


class AnalyzeDataset(Tool):
    name = "analyze_dataset"
    description = ("Deterministic statistics on a CSV dataset of the knowledge base (pandas, allowlisted operations): "
                   "row_count, describe_columns, missing_values, mean, median, min, max, sum, std, var, quantile, count, "
                   "value_counts, groupby_agg, correlation, select_rows, lookup_value; optional filters. "
                   "std/var are SAMPLE statistics (ddof=1) unless ddof=0.")
    Input = AnalyzeInput
    output_description = "value | table + provenance (dataset id, file hash, rows used); citable evidence id D#"
    examples = [{"dataset_id": "product_prices", "operation": "median", "column": "price_usd"},
                {"dataset_id": "ml_model_evaluation_runs", "operation": "lookup_value", "column": "val_accuracy",
                 "filters": [{"column": "run_id", "op": "==", "value": "ML004"}]}]

    def run(self, a: AnalyzeInput, ctx: ToolContext):
        path = ctx.kb.dataset_path(a.dataset_id, scope_courses(ctx, None))
        if path is None:
            ids = [d["dataset_id"] for d in ctx.kb.dataset_catalog(scope_courses(ctx, None))]
            raise ToolError(f"dataset '{a.dataset_id}' not found in scope. Available: {ids}")
        raw = path.read_bytes()
        df = pd.read_csv(path)
        doc = ctx.kb.get_document(a.dataset_id)
        total = len(df)
        op = a.operation
        if op == "describe_columns":
            val = {"table": [{"column": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum())} for c in df.columns]}
            stmt = f"{a.dataset_id} has {total} rows and columns {list(df.columns)}"
            used = total
        else:
            df = apply_filters(df, a.filters)
            used = len(df)
            if used == 0 and op != "row_count":
                raise ToolError("no rows match the filters")
            val, stmt = self._compute(df, a, total)
        fdesc = " where " + " and ".join(f"{f.column} {f.op} {f.value}" for f in a.filters) if a.filters else ""
        stmt = f"[{a.dataset_id}] {stmt}{fdesc} (rows used: {used} of {total})"
        data = {"dataset_id": a.dataset_id, "operation": op, "column": a.column, "rows_used": used, "rows_total": total,
                "filters": [f.model_dump() for f in a.filters], "ddof": a.ddof if op in ("std", "var") else None,
                "file": doc["filename"] if doc else path.name, "file_sha256": sha256_bytes(raw)[:16], **val}
        it = ctx.evidence.add_computation("dataset", f"Dataset analysis: {a.dataset_id}", stmt, data,
                                          doc_id=a.dataset_id, course_id=doc["course_id"] if doc else None,
                                          location=f"{op}" + (f" on {a.column}" if a.column else ""))
        return data, [it.id], stmt

    def _compute(self, df: pd.DataFrame, a: AnalyzeInput, total: int):
        op = a.operation
        if op == "row_count":
            return {"value": len(df)}, f"row count = {len(df)}"
        if op == "missing_values":
            m = df.isna().sum()
            return {"table": [{"column": c, "missing": int(m[c])} for c in df.columns], "value": int(m.sum())}, f"total missing cells = {int(m.sum())}"
        if op in ("mean", "median", "min", "max", "sum", "std", "var", "quantile"):
            c = _col(df, a.column)
            s = _num(df, c)
            v = {"mean": s.mean, "median": s.median, "min": s.min, "max": s.max, "sum": s.sum,
                 "std": lambda: s.std(ddof=a.ddof), "var": lambda: s.var(ddof=a.ddof), "quantile": lambda: s.quantile(a.q)}[op]()
            if pd.isna(v):
                raise ToolError("result undefined (too few values)")
            label = {"std": f"{'sample' if a.ddof else 'population'} standard deviation", "var": f"{'sample' if a.ddof else 'population'} variance",
                     "quantile": f"quantile q={a.q}"}.get(op, op)
            return {"value": _py(v), "column": c}, f"{label} of '{c}' = {_py(v)}"
        if op == "count":
            c = _col(df, a.column)
            return {"value": int(df[c].notna().sum())}, f"count of non-missing '{c}' = {int(df[c].notna().sum())}"
        if op == "value_counts":
            c = _col(df, a.column)
            vc = df[c].value_counts().head(a.limit)
            return {"table": [{"value": str(k), "count": int(v)} for k, v in vc.items()]}, f"value counts of '{c}': " + ", ".join(f"{k}={v}" for k, v in vc.items())
        if op == "groupby_agg":
            g, c = _col(df, a.group_by, "group_by"), _col(df, a.column)
            agg = a.agg or "mean"
            if agg != "count":
                _num(df, c)
            r = df.groupby(g)[c].agg(agg).sort_index()
            return {"table": [{g: str(k), agg: _py(v)} for k, v in r.items()]}, f"{agg} of '{c}' by '{g}': " + ", ".join(f"{k}={_py(v)}" for k, v in r.items())
        if op == "correlation":
            c1, c2 = _col(df, a.column), _col(df, a.other_column, "other_column")
            _num(df, c1), _num(df, c2)
            if len(df) < 3:
                raise ToolError("need at least 3 rows for a correlation")
            v = df[c1].corr(df[c2])
            if pd.isna(v):
                raise ToolError("correlation undefined (constant column)")
            return {"value": _py(v)}, f"Pearson correlation of '{c1}' and '{c2}' = {_py(v)}"
        if op == "select_rows":
            r = df.head(a.limit)
            return {"table": [{k: _py(v) for k, v in row.items()} for row in r.to_dict("records")]}, f"{len(r)} rows selected: " + " | ".join(
                ", ".join(f"{k}={_py(v)}" for k, v in row.items()) for row in r.to_dict("records")[:3])
        if op == "lookup_value":
            c = _col(df, a.column)
            if len(df) != 1:
                raise ToolError(f"lookup needs exactly 1 matching row, found {len(df)}; add filters")
            v = _py(df.iloc[0][c])
            row = {k: _py(x) for k, x in df.iloc[0].to_dict().items()}
            return {"value": v, "row": row}, f"'{c}' = {v} (row: " + ", ".join(f"{k}={x}" for k, x in list(row.items())[:4]) + ")"
        raise ToolError("unsupported operation")
