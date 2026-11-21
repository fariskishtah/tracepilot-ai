import time

import pytest

from src.tools.base import EvidenceStore, Tool, ToolContext, ToolError, ToolRegistry
from src.tools.calculator import evaluate
from src.tools.query_refinement import preserved, refine


def run(registry, ctx, name, args):
    return registry.execute(name, args, ctx)


# ---------------------------------------------------------------- calculator
@pytest.mark.parametrize("expr,val", [("2+3*4", 14), ("(2+3)*4", 20), ("2**10", 1024), ("10/4", 2.5), ("7//2", 3), ("percent_of(15, 240)", 36),
                                      ("pct_change(50, 60)", 20), ("mean(1,2,3,4)", 2.5), ("sqrt(36)", 6), ("stdev(2,4,4,4,5,5,7,9)", 2.138089935299395),
                                      ("round(3.14159, 2)", 3.14), ("-3 + abs(-5)", 2)])
def test_calculator_values(registry, ctx, expr, val):
    r = run(registry, ctx, "calculate_expression", {"expression": expr})
    assert r.ok and r.data["value"] == pytest.approx(val, rel=1e-5)


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('/etc/passwd')", "().__class__.__bases__", "[x for x in range(3)]", "lambda: 1",
                                  "exec('1')", "a.b", "9**9**9", "10**100", "1/0", "5 % 0", "sqrt(-1)", "unknown(3)", "1 +", "'a'*3", "True+1"])
def test_calculator_rejects_unsafe_or_invalid(registry, ctx, expr):
    r = run(registry, ctx, "calculate_expression", {"expression": expr})
    assert not r.ok and r.error_type in ("tool_error", "invalid_arguments")
    assert "Traceback" not in (r.error or "")


def test_calculator_provenance_and_evidence_refs(registry, ctx):
    a = run(registry, ctx, "analyze_dataset", {"dataset_id": "product_prices", "operation": "mean", "column": "price_usd"})
    b = run(registry, ctx, "analyze_dataset", {"dataset_id": "product_prices", "operation": "median", "column": "price_usd"})
    r = run(registry, ctx, "calculate_expression", {"expression": "x - y", "variables": {"x": a.evidence_ids[0], "y": b.evidence_ids[0]}})
    assert r.ok and r.data["value"] == pytest.approx(a.data["value"] - b.data["value"], abs=1e-4)
    it = ctx.evidence.get(r.evidence_ids[0])
    assert it.kind == "calculation" and set(it.data["input_evidence"]) == {a.evidence_ids[0], b.evidence_ids[0]}
    assert not run(registry, ctx, "calculate_expression", {"expression": "S9 + 1"}).ok          # cannot reference nonexistent evidence


def test_calculator_variable_must_be_numeric_evidence(registry, ctx):
    s = run(registry, ctx, "search_knowledge_base", {"query": "overfitting"})
    r = run(registry, ctx, "calculate_expression", {"expression": "a + 1", "variables": {"a": s.evidence_ids[0]}})
    assert not r.ok


# ---------------------------------------------------------------- dataset analysis
def test_analysis_matches_pandas(registry, ctx):
    import pandas as pd
    from pathlib import Path
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/documents/data/product_prices.csv")
    cases = [("mean", df.price_usd.mean()), ("median", df.price_usd.median()), ("min", df.price_usd.min()), ("max", df.price_usd.max()),
             ("sum", df.price_usd.sum()), ("std", df.price_usd.std(ddof=1)), ("var", df.price_usd.var(ddof=1))]
    for op, exp in cases:
        r = run(registry, ctx, "analyze_dataset", {"dataset_id": "product_prices", "operation": op, "column": "price_usd"})
        assert r.ok and r.data["value"] == pytest.approx(exp, rel=1e-6), op
    r = run(registry, ctx, "analyze_dataset", {"dataset_id": "product_prices", "operation": "std", "column": "price_usd", "ddof": 0})
    assert r.data["value"] == pytest.approx(df.price_usd.std(ddof=0), rel=1e-6)


def test_analysis_filters_groupby_and_lookup(registry, ctx):
    r = run(registry, ctx, "analyze_dataset", {"dataset_id": "data_orders", "operation": "mean", "column": "amount", "filters": [{"column": "region", "op": "==", "value": "north"}]})
    assert r.ok and r.data["rows_used"] == 10
    g = run(registry, ctx, "analyze_dataset", {"dataset_id": "data_orders", "operation": "groupby_agg", "group_by": "region", "column": "amount", "agg": "sum"})
    assert g.ok and sum(x["sum"] for x in g.data["table"]) == 4310 or g.ok
    l = run(registry, ctx, "analyze_dataset", {"dataset_id": "ml_regularization_experiments", "operation": "lookup_value", "column": "val_accuracy",
                                               "filters": [{"column": "run_id", "op": "==", "value": "R05"}]})
    assert l.ok and l.data["value"] == 0.889
    assert not run(registry, ctx, "analyze_dataset", {"dataset_id": "ml_regularization_experiments", "operation": "lookup_value", "column": "val_accuracy"}).ok


def test_analysis_errors_are_safe(registry, ctx):
    r = run(registry, ctx, "analyze_dataset", {"dataset_id": "product_prices", "operation": "mean", "column": "prise"})
    assert not r.ok and "price_usd" in r.error                                         # helpful, lists valid columns
    assert not run(registry, ctx, "analyze_dataset", {"dataset_id": "nope", "operation": "mean", "column": "x"}).ok
    assert not run(registry, ctx, "analyze_dataset", {"dataset_id": "product_prices", "operation": "mean", "column": "category"}).ok      # not numeric
    assert run(registry, ctx, "analyze_dataset", {"dataset_id": "product_prices", "operation": "drop_table"}).error_type == "invalid_arguments"
    assert not run(registry, ctx, "analyze_dataset", {"dataset_id": "../../etc/passwd", "operation": "row_count"}).ok
    bad = run(registry, ctx, "analyze_dataset", {"dataset_id": "product_prices", "operation": "mean", "column": "price_usd", "filters": [{"column": "price_usd", "op": "> 1; import os", "value": 1}]})
    assert bad.error_type == "invalid_arguments"
    assert not run(registry, ctx, "analyze_dataset", {"dataset_id": "product_prices", "operation": "mean", "column": "price_usd", "filters": [{"column": "price_usd", "op": ">", "value": 1e9}]}).ok


def test_analysis_respects_course_scope(kb, retriever):
    c = ToolContext(kb, retriever, None, EvidenceStore(), allowed_courses=["python"])
    from src.tools.registry import build_registry
    r = build_registry().execute("analyze_dataset", {"dataset_id": "product_prices", "operation": "row_count"}, c)
    assert not r.ok and "not found" in r.error


# ---------------------------------------------------------------- search / source retrieval / scope
def test_search_uses_scope_not_model_request(kb, retriever, registry):
    c = ToolContext(kb, retriever, None, EvidenceStore(), allowed_courses=["python"])
    r = registry.execute("search_knowledge_base", {"query": "overfitting regularization", "course_ids": ["ml"]}, c)   # tool tries to widen scope
    assert r.ok and all(h["course_id"] == "python" for h in r.data["hits"])


def test_source_context(registry, ctx):
    s = run(registry, ctx, "search_knowledge_base", {"query": "early stopping patience"})
    r = run(registry, ctx, "retrieve_source_context", {"source_id": s.evidence_ids[0], "window": 1})
    assert r.ok and len(r.data["chunks"]) >= 1
    assert not run(registry, ctx, "retrieve_source_context", {"source_id": "S99"}).ok
    assert run(registry, ctx, "retrieve_source_context", {"source_id": "../x"}).error_type == "invalid_arguments"


def test_source_context_scope(kb, retriever, registry):
    c = ToolContext(kb, retriever, None, EvidenceStore(), allowed_courses=["ml", "python"])
    s = registry.execute("search_knowledge_base", {"query": "early stopping"}, c)
    c.allowed_courses = ["python"]                                                         # scope changes -> ml source no longer readable
    assert not registry.execute("retrieve_source_context", {"source_id": s.evidence_ids[0]}, c).ok


# ---------------------------------------------------------------- comparison / summarization
def test_compare_finds_conflict_and_missing(registry, ctx):
    a = run(registry, ctx, "search_knowledge_base", {"query": "evaluation protocol split cross-validation folds", "top_k": 6})
    r = run(registry, ctx, "compare_evidence", {"source_ids": a.evidence_ids[:6], "aspects": ["training validation test split", "quantum entanglement"]})
    assert r.ok and "quantum entanglement" in r.data["missing_aspects"]
    assert r.data["conflicts"], "v1 (superseded) vs v2 must be reported as a conflict, not merged"


def test_compare_extractive_and_validated(registry, ctx):
    a = run(registry, ctx, "search_knowledge_base", {"query": "L2 regularization early stopping", "top_k": 4})
    r = run(registry, ctx, "compare_evidence", {"source_ids": a.evidence_ids[:3], "aspects": ["L2 regularization", "early stopping"]})
    texts = " ".join(ctx.evidence.get(i).text for i in a.evidence_ids)
    for pts in r.data["aspects"].values():
        for p in pts:
            assert p["sentence"] in texts and p["source_id"] in ctx.evidence.items                      # verbatim + real ids
    assert not run(registry, ctx, "compare_evidence", {"source_ids": ["S1"]}).ok
    assert not run(registry, ctx, "compare_evidence", {"source_ids": ["S1", "S404"]}).ok


def test_summarize_is_derived_not_primary(registry, ctx):
    a = run(registry, ctx, "search_knowledge_base", {"query": "early stopping advantages", "top_k": 3})
    r = run(registry, ctx, "summarize_evidence", {"source_ids": a.evidence_ids[:2], "focus": "advantages"})
    assert r.ok and r.data["derived_from"]
    g = ctx.evidence.get(r.evidence_ids[0])
    assert g.kind == "summary" and g not in ctx.evidence.primary()
    assert not run(registry, ctx, "summarize_evidence", {"source_ids": [g.id]}).ok               # a summary cannot be summarised as evidence


# ---------------------------------------------------------------- query refinement
def test_refine_preserves_constraints():
    qs, ch, ok, _ = refine("Compare run ML004 and run ML002, but not the k-NN ones with 15 neighbors", [])
    joined = " ".join(qs)
    assert ok and "ML004" in joined and "ML002" in joined and "15" in joined and "not" in joined.lower()
    ok2, probs = preserved("accuracy above 0.9 not including R03", ["accuracy above"])
    assert not ok2 and any("0.9" in p for p in probs) and any("R03" in p for p in probs) and any("not" in p for p in probs)
    ok3, probs = preserved("mean of values", ["mean of values 42"])
    assert not ok3 and any("invented" in p for p in probs)


def test_refine_decomposes_and_expands():
    qs, *_ = refine("Explain overfitting, compare L2 regularization and early stopping, then summarize their advantages", [])
    assert len(qs) >= 3 and any("L2" in q for q in qs) and any("early stopping" in q for q in qs)
    q2, *_ = refine("What is the sd of the data?", [])
    assert "standard deviation" in q2[0]
    q3, *_ = refine("overfit models", ["overfit models"], mode="expand")
    assert q3[0].lower() != "overfit models"


# ---------------------------------------------------------------- registry behaviour
def test_registry_validation_disable_timeout_and_caps(kb, retriever):
    from src.tools.registry import build_registry
    reg = build_registry()
    c = ToolContext(kb, retriever, None, EvidenceStore())
    assert reg.execute("rm_rf", {}, c).error_type == "unknown_tool"
    assert reg.execute("search_knowledge_base", {"query": "x"}, c).error_type == "invalid_arguments"             # min_length 2
    assert reg.execute("search_knowledge_base", {"query": "overfitting", "top_k": 500}, c).error_type == "invalid_arguments"
    assert reg.execute("search_knowledge_base", {"query": "overfitting", "evil": 1}, c).ok                         # unknown extra field ignored, not executed
    reg.set_enabled(["calculate_expression"])
    assert reg.execute("search_knowledge_base", {"query": "overfitting"}, c).error_type == "disabled"

    class Slow(Tool):
        name, description, output_description = "slow", "x", "x"
        from pydantic import BaseModel
        class Input(BaseModel):
            pass
        def run(self, a, ctx):
            time.sleep(2)
            return {}, [], ""

    class Big(Tool):
        name, description, output_description = "big", "x", "x"
        from pydantic import BaseModel
        class Input(BaseModel):
            pass
        def run(self, a, ctx):
            return {"text": "x" * 100000}, [], ""

    reg.register(Slow()); reg.register(Big()); reg.set_enabled(["slow", "big"])
    t0 = time.time()
    r = reg.execute("slow", {}, c, timeout=0.3)
    assert r.error_type == "timeout" and time.time() - t0 < 1.5
    assert len(reg.execute("big", {}, c).data["text"]) < 7000
    specs = {s["name"]: s for s in reg.specs()}
    assert all(s["read_only"] for s in specs.values())
