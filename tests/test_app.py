import json

from src.ui import render as R


def test_timeline_and_workflow_render_real_trace(kb, retriever):
    from tests.conftest import FakeLLM, make_agent
    plan = {"question_type": "comparison", "complexity": "moderate", "requires_retrieval": True, "requires_calculation": False, "requires_data_analysis": False, "requires_comparison": True,
            "subtasks": [{"id": "t1", "tool": "search_knowledge_base", "goal": "a", "query": "supervised learning labels"}, {"id": "t2", "tool": "search_knowledge_base", "goal": "b", "query": "unsupervised learning clustering"},
                         {"id": "t3", "tool": "compare_evidence", "goal": "compare", "depends_on": ["t1", "t2"], "aspects": ["supervised", "unsupervised"]}]}
    res = make_agent(kb, retriever, FakeLLM(plan=plan)).run("Compare supervised and unsupervised learning.").to_dict()
    html = R.timeline_html(res["trace"])
    assert "tool selected" in html and "compare_evidence" in html and "plan" in html
    svg = R.workflow_svg(res)
    assert "<svg" in svg and "compare_evidence" in svg and "t1" in svg
    rows = R.evidence_table(res)
    assert rows and rows[0][2] in ("original document", "dataset computation", "calculation", "generated summary")
    assert "tool calls" in R.metrics_md(res)
    json.dumps(res, default=str)                                                           # trace is serialisable


def test_empty_states():
    assert "No execution yet" in R.timeline_html([]) and "Run a question" in R.workflow_svg(None)


def test_build_app_constructs(monkeypatch, kb, retriever):
    from src.service import Service
    from src.ui.app_ui import build_app
    svc = Service.__new__(Service)
    from tests.conftest import FakeLLM
    from src.agents.baseline import StandardRAG
    import threading
    svc.kb, svc.retriever, svc.llm, svc._agents, svc.lock = kb, retriever, FakeLLM(), {}, threading.Lock()
    svc.baseline = StandardRAG(kb, retriever, svc.llm)
    app = build_app(svc)
    assert app is not None
