"""Safe calculator: AST allowlist, no eval/exec, bounded size, deterministic results."""
from __future__ import annotations

import ast
import math
import operator as op
import statistics as st

from pydantic import BaseModel, Field

from src.tools.base import Tool, ToolContext, ToolError

BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
       ast.Mod: op.mod, ast.Pow: op.pow}
UN = {ast.USub: op.neg, ast.UAdd: op.pos}
MAX_ABS, MAX_EXP, MAX_NODES = 1e15, 12, 80


def _std(*a):
    return st.stdev(_flat(a))


def _flat(a):
    return [float(x) for x in (a[0] if len(a) == 1 and isinstance(a[0], (list, tuple)) else a)]


FUNCS = {
    "abs": abs, "round": lambda x, n=0: round(x, int(n)), "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "exp": math.exp, "min": lambda *a: min(_flat(a)), "max": lambda *a: max(_flat(a)),
    "sum": lambda *a: sum(_flat(a)), "mean": lambda *a: st.fmean(_flat(a)), "median": lambda *a: st.median(_flat(a)),
    "stdev": _std, "pstdev": lambda *a: st.pstdev(_flat(a)), "var": lambda *a: st.variance(_flat(a)),
    "pvar": lambda *a: st.pvariance(_flat(a)),
    "percent_of": lambda pct, x: pct / 100.0 * x,            # percent_of(20, 150) -> 30
    "pct_change": lambda old, new: (new - old) / old * 100,   # pct_change(50, 60) -> 20
    "pct_diff": lambda a, b: abs(a - b) / ((a + b) / 2) * 100,
}
CONST = {"pi": math.pi, "e": math.e}


class CalcInput(BaseModel):
    expression: str = Field(min_length=1, max_length=200)
    variables: dict[str, float | str] = Field(default_factory=dict, max_length=10,
                                              description="name -> number, or name -> evidence id (D1/C1) holding a numeric result")


def evaluate(expr: str, names: dict[str, float]) -> float:
    try:
        tree = ast.parse(expr.strip().replace("^", "**").replace("×", "*").replace("÷", "/"), mode="eval")
    except (SyntaxError, ValueError):
        raise ToolError("invalid expression syntax")
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ToolError("expression too complex")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in names:
                return names[n.id]
            if n.id in CONST:
                return CONST[n.id]
            raise ToolError(f"unknown name '{n.id}'")
        if isinstance(n, ast.BinOp) and type(n.op) in BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > MAX_EXP or abs(a) > 1e6):
                raise ToolError("exponent too large")
            if isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0:
                raise ToolError("division by zero")
            return BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in UN:
            return UN[type(n.op)](ev(n.operand))
        if isinstance(n, (ast.List, ast.Tuple)):
            return [ev(x) for x in n.elts]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords:
            try:
                return FUNCS[n.func.id](*[ev(x) for x in n.args])
            except ToolError:
                raise
            except (ValueError, ZeroDivisionError, TypeError, st.StatisticsError, OverflowError) as e:
                raise ToolError(f"math error in {n.func.id}: {type(e).__name__}")
        raise ToolError(f"unsupported operation: {type(n).__name__}")

    r = ev(tree)
    if isinstance(r, list) or not isinstance(r, (int, float)) or math.isnan(r) or math.isinf(r) or abs(r) > MAX_ABS:
        raise ToolError("result is not a finite bounded number")
    return float(r)


class CalculateExpression(Tool):
    name = "calculate_expression"
    description = ("Deterministic arithmetic: + - * / ** %, abs, round, sqrt, log, mean, median, stdev, var, percent_of, "
                   "pct_change. Variables may be numbers or evidence ids of earlier numeric results (e.g. 'D2 - D1').")
    Input = CalcInput
    output_description = "value (float), expression, resolved variables; provenance evidence id C#"
    examples = [{"expression": "D2 - D1"}, {"expression": "pct_change(0.81, 0.89)"}]

    def run(self, a: CalcInput, ctx: ToolContext):
        import re
        names: dict[str, float] = {}
        used: list[str] = []
        for k, v in a.variables.items():
            if not re.fullmatch(r"[A-Za-z_]\w{0,20}", k):
                raise ToolError(f"bad variable name '{k}'")
            if isinstance(v, str):
                num = ctx.evidence.numeric(v)
                if num is None:
                    raise ToolError(f"'{v}' is not a numeric result")
                names[k] = num
                used.append(v)
            else:
                names[k] = float(v)
        for ref in re.findall(r"\b[DC]\d+\b", a.expression):   # evidence ids directly in the expression
            if ref not in names:
                num = ctx.evidence.numeric(ref)
                if num is None:
                    raise ToolError(f"'{ref}' is not a numeric result")
                names[ref] = num
                used.append(ref)
        val = evaluate(a.expression, names)
        shown = round(val, 6)
        it = ctx.evidence.add_computation(
            "calculation", "Calculator", f"{a.expression.strip()} = {shown:g}"
            + (f" (inputs: " + ", ".join(f"{k}={v:g}" for k, v in names.items()) + ")" if names else ""),
            {"value": shown, "expression": a.expression, "inputs": names, "input_evidence": sorted(set(used))})
        return {"expression": a.expression, "value": shown, "inputs": names}, [it.id], f"{a.expression} = {shown:g}"
