"""State-machine logic with a deterministic fake LLM (real retrieval/tools)."""
import pytest

from tests.conftest import FakeLLM, make_agent

SEARCH = {"question_type": "factual", "complexity": "simple", "requires_retrieval": True, "requires_calculation": False, "requires_data_analysis": False,
          "requires_comparison": False, "subtasks": [{"id": "t1", "tool": "search_knowledge_base", "goal": "overfitting definition", "query": "overfitting definition"}]}


def types(res):
    return [e["type"] for e in res.trace]


def test_transitions_and_trace_fields(kb, retriever):
    res = make_agent(kb, retriever, FakeLLM(plan=SEARCH)).run("What is overfitting?")
    t = types(res)
    order = ["request_received", "question_analysis", "plan_created", "tool_selected", "tool_result", "validation", "synthesis_started", "citation_check", "answer", "completed"]
    pos = [t.index(x) for x in order]
    assert pos == sorted(pos)
    assert res.stop_reason == "answered" and res.tool_sequence == ["search_knowledge_base"] and not res.abstained
    assert res.metrics["tool_calls"] == 1 and res.metrics["latency_s"] > 0 and res.plan["subtasks"]
    assert all({"seq", "t_ms", "type"} <= e.keys() for e in res.trace)
    assert res.citations and all(c["id"] in res.evidence for c in res.citations)


def test_replans_weak_evidence_then_abstains(kb, retriever):
    plan = dict(SEARCH, subtasks=[{"id": "t1", "tool": "search_knowledge_base", "goal": "find", "query": "quantum chromodynamics gluon plasma"}])
    res = make_agent(kb, retriever, FakeLLM(plan=plan)).run("quantum chromodynamics gluon plasma?")
    assert res.abstained and res.stop_reason == "insufficient_evidence"
    assert res.metrics["tool_calls"] == 3 and res.metrics["replans"] == 2                       # bounded: 1 + 2 retries
    queries = [e["args"]["query"] for e in res.trace if e["type"] == "tool_selected"]
    assert len(set(queries)) == 3, "each retry must use a genuinely different query"
    assert "abstain" in types(res) and "synthesis_started" not in types(res)


def test_no_replan_when_disabled(kb, retriever):
    plan = dict(SEARCH, subtasks=[{"id": "t1", "tool": "search_knowledge_base", "goal": "find", "query": "quantum chromodynamics gluon plasma"}])
    res = make_agent(kb, retriever, FakeLLM(plan=plan), replan=False).run("quantum chromodynamics gluon plasma?")
    assert res.metrics["tool_calls"] == 1 and res.metrics["replans"] == 0 and "replan_skipped" in types(res)


def test_tool_budget_enforced(kb, retriever):
    plan = dict(SEARCH, subtasks=[{"id": f"t{i}", "tool": "search_knowledge_base", "goal": f"g{i}", "query": f"overfitting {w}"} for i, w in enumerate(["a", "b", "c", "d", "e", "f", "g"], 1)])
    from src.config import cfg
    res = make_agent(kb, retriever, FakeLLM(plan=plan)).run("What is overfitting and many things?")
    assert res.metrics["tool_calls"] <= cfg("agent")["budgets"]["max_tool_calls"]


def test_budget_exhaustion_stop_reason(kb, retriever, monkeypatch):
    import src.agents.controller as C
    real = C.cfg
    def tiny(name):
        d = real(name)
        if name == "agent":
            d = dict(d, budgets=dict(d["budgets"], max_tool_calls=1, max_plan_steps=6))
        return d
    monkeypatch.setattr(C, "cfg", tiny)
    plan = dict(SEARCH, subtasks=[{"id": "t1", "tool": "search_knowledge_base", "goal": "a", "query": "overfitting"}, {"id": "t2", "tool": "search_knowledge_base", "goal": "b", "query": "regularization"}])
    res = make_agent(kb, retriever, FakeLLM(plan=plan)).run("overfitting and regularization")
    assert res.stop_reason == "tool_budget_exhausted" and res.metrics["tool_calls"] == 1 and "budget_exhausted" in types(res)


def test_evidence_accumulates_and_calculation_chain(kb, retriever):
    plan = {"question_type": "calculation", "complexity": "moderate", "requires_retrieval": False, "requires_calculation": True, "requires_data_analysis": True, "requires_comparison": False,
            "subtasks": [{"id": "t1", "tool": "analyze_dataset", "goal": "r02", "dataset_id": "ml_regularization_experiments", "operation": "lookup_value", "column": "val_accuracy", "filters": [{"column": "run_id", "op": "==", "value": "R02"}]},
                         {"id": "t2", "tool": "analyze_dataset", "goal": "r04", "dataset_id": "ml_regularization_experiments", "operation": "lookup_value", "column": "val_accuracy", "filters": [{"column": "run_id", "op": "==", "value": "R04"}]},
                         {"id": "t3", "tool": "calculate_expression", "goal": "diff", "expression": "t1 - t2", "depends_on": ["t1", "t2"]}]}
    res = make_agent(kb, retriever, FakeLLM(plan=plan)).run("Calculate the difference between the val_accuracy of run R02 and run R04 in the regularization experiments dataset.")
    assert res.tool_sequence == ["analyze_dataset", "analyze_dataset", "calculate_expression"]
    assert "0.008" in res.answer and "[C1]" in res.answer                                  # deterministic numbers always present
    assert {"D1", "D2", "C1"} <= set(res.evidence)


def test_failed_dependency_skips_calculation(kb, retriever):
    plan = {"question_type": "calculation", "complexity": "moderate", "requires_retrieval": False, "requires_calculation": True, "requires_data_analysis": True, "requires_comparison": False,
            "subtasks": [{"id": "t1", "tool": "analyze_dataset", "goal": "x", "dataset_id": "product_prices", "operation": "mean", "column": "price_usd", "filters": [{"column": "price_usd", "op": ">", "value": 1000000}]},
                         {"id": "t2", "tool": "calculate_expression", "goal": "x", "expression": "t1 + 1", "depends_on": ["t1"]}]}
    res = make_agent(kb, retriever, FakeLLM(plan=plan), replan=False).run("mean price_usd of products above one million in the product prices dataset")
    assert "calculate_expression" not in res.tool_sequence and "step_skipped" in types(res) and res.abstained


def test_recoverable_error_is_repaired(kb, retriever):
    plan = {"question_type": "data_analysis", "complexity": "simple", "requires_retrieval": False, "requires_calculation": False, "requires_data_analysis": True, "requires_comparison": False,
            "subtasks": [{"id": "t1", "tool": "analyze_dataset", "goal": "mean price", "dataset_id": "product_prices", "operation": "mean", "column": "price"}]}
    res = make_agent(kb, retriever, FakeLLM(plan=plan)).run("What is the mean price_usd in the product prices dataset?")
    assert res.metrics["replans"] == 1 and res.metrics["tool_calls"] == 2
    assert "105.04" in res.answer and any(e["type"] == "replan" for e in res.trace)


def test_scope_isolation_end_to_end(kb, retriever):
    res = make_agent(kb, retriever, FakeLLM(plan=SEARCH)).run("What is overfitting?", ["python"])
    assert all(e["course_id"] == "python" for e in res.evidence.values() if e["kind"] == "document")


def test_invalid_questions_and_llm_down(kb, retriever):
    a = make_agent(kb, retriever, FakeLLM(plan=SEARCH))
    assert a.run("").stop_reason == "error"
    assert a.run("x" * 2000).stop_reason == "error"
    down = FakeLLM(plan=SEARCH)
    down.chat = lambda *a, **k: (_ for _ in ()).throw(__import__("src.llm", fromlist=["LLMError"]).LLMError("down"))
    r = make_agent(kb, retriever, down).run("What is overfitting?")
    assert r.stop_reason == "llm_unavailable" and r.abstained


def test_standard_rag_baseline_single_retrieval(kb, retriever):
    from src.agents.baseline import StandardRAG
    r = StandardRAG(kb, retriever, FakeLLM()).run("What is overfitting?")
    assert r.system == "standard_rag" and r.metrics["tool_calls"] == 1 and r.metrics["llm_calls"] >= 1 and r.plan is None


def test_llm_budget_caps_optional_llm_steps(kb, retriever, monkeypatch):
    import src.agents.controller as C
    real = C.cfg
    monkeypatch.setattr(C, "cfg", lambda n: dict(real(n), budgets=dict(real(n)["budgets"], max_llm_calls=0)) if n == "agent" else real(n))
    plan = dict(SEARCH, subtasks=[{"id": "t1", "tool": "search_knowledge_base", "goal": "a", "query": "early stopping"},
                                  {"id": "t2", "tool": "summarize_evidence", "goal": "advantages", "depends_on": ["t1"]}])
    res = make_agent(kb, retriever, FakeLLM(plan=plan)).run("Summarize early stopping")
    assert "summarize_evidence" not in res.tool_sequence and any(e["type"] == "step_skipped" and e.get("reason") == "LLM call budget reached" for e in res.trace)
