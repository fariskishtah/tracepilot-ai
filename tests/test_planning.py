import pytest

from src.agents.planner import Planner
from src.agents.routing import analyze_question, guard_plan, pick_column, pick_dataset, rule_plan
from src.agents.state import QuestionPlan, Subtask
from tests.conftest import FakeLLM

ALL = ["search_knowledge_base", "retrieve_source_context", "summarize_evidence", "analyze_dataset", "calculate_expression", "compare_evidence", "refine_search_query"]


@pytest.fixture(scope="module")
def cat(kb):
    return kb.dataset_catalog()


def plan(q, cat, enabled=ALL):
    f = analyze_question(q, cat)
    return rule_plan(q, cat, f, enabled, None), f


def tools(p):
    return [s.tool for s in p.subtasks]


def test_features(cat):
    f = analyze_question("Calculate the mean of price_usd in the product prices dataset", cat)
    assert f["calc_cue"] and f["data_cue"] and f["needs_data"] and "product_prices" in f["named_datasets"]
    g = analyze_question("What is supervised learning?", cat)
    assert not g["needs_data"] and not g["compare_cue"] and g["concept_cue"]
    assert analyze_question("Compare A and B", cat)["compare_cue"]


def test_different_questions_different_paths(cat):
    simple, _ = plan("What is overfitting?", cat)
    comp, _ = plan("Compare supervised and unsupervised learning.", cat)
    num, _ = plan("Calculate the mean of price_usd in the product prices dataset", cat)
    assert tools(simple) == ["search_knowledge_base"]
    assert tools(comp) == ["search_knowledge_base", "search_knowledge_base", "compare_evidence"]
    assert tools(num)[0] == "analyze_dataset" and "compare_evidence" not in tools(num)
    assert len({tuple(tools(x)) for x in (simple, comp, num)}) == 3


def test_numeric_question_has_no_pointless_search_and_picks_column(cat):
    p, _ = plan("What is the median price_usd in the product prices dataset?", cat)
    s = p.subtasks[0]
    assert tools(p) == ["analyze_dataset"] and s.operation == "median" and s.column == "price_usd" and s.dataset_id == "product_prices"


def test_multi_part_decomposition(cat):
    p, _ = plan("Compare the average and median price in the product prices dataset, then explain why they differ.", cat)
    assert tools(p).count("analyze_dataset") == 2 and "search_knowledge_base" in tools(p)


def test_dataset_and_column_picking(cat):
    assert pick_dataset("validation accuracy of the regularization experiments", cat)["dataset_id"] == "ml_regularization_experiments"
    assert pick_dataset("order amounts by region", cat)["dataset_id"] == "data_orders"
    assert pick_column("validation accuracy", ["run_id", "train_accuracy", "val_accuracy"]) == "val_accuracy"


def test_disabled_tools_never_planned(cat):
    p, _ = plan("Calculate the mean of price_usd in the product prices dataset", cat, ["search_knowledge_base"])
    assert set(tools(p)) == {"search_knowledge_base"}


def test_guard_repairs_bad_llm_plans(cat):
    q = "Compare the average and median price in the product prices dataset, then explain why they differ."
    f = analyze_question(q, cat)
    bad = QuestionPlan(question_type="multi_step", complexity="complex", requires_retrieval=False, requires_calculation=True, requires_data_analysis=True, requires_comparison=True,
                       subtasks=[Subtask(id="t1", tool="analyze_dataset", goal="mean", dataset_id="product_prices", operation="mean", column="price_usd"),
                                 Subtask(id="t2", tool="shell_exec", goal="rm -rf /"),
                                 Subtask(id="t3", tool="compare_evidence", goal="cmp", depends_on=["t1"]),
                                 Subtask(id="t4", tool="calculate_expression", goal="calc", expression="foo - bar", depends_on=["t1"]),
                                 Subtask(id="t5", tool="analyze_dataset", goal="x", dataset_id="product_prices", operation="groupby_agg")])
    g, adj = guard_plan(bad, q, f, ALL, cat, 7)
    ts = tools(g)
    assert "shell_exec" not in ts and "compare_evidence" not in ts           # unknown tool + compare without retrieval removed
    assert not any(s.operation == "groupby_agg" for s in g.subtasks)         # non-executable step removed
    assert "search_knowledge_base" in ts                                     # explanation clause got retrieval
    assert adj


def test_guard_dag_order_and_limits(cat):
    f = analyze_question("What is overfitting?", cat)
    p = QuestionPlan(question_type="factual", complexity="simple", requires_retrieval=True, requires_calculation=False, requires_data_analysis=False, requires_comparison=False,
                     subtasks=[Subtask(id="t1", tool="compare_evidence", goal="x", depends_on=["t2"]), Subtask(id="t2", tool="search_knowledge_base", goal="s", query="overfitting")]
                     + [Subtask(id=f"t{i}", tool="search_knowledge_base", goal="s", query=f"q{i}") for i in range(3, 9)])
    g, _ = guard_plan(p, "What is overfitting?", f, ALL, cat, 5)
    assert len(g.subtasks) <= 5 and g.subtasks[0].tool == "search_knowledge_base"


def test_llm_planner_valid_and_fallback(cat):
    q = "What is overfitting?"
    f = analyze_question(q, cat)
    good = {"question_type": "factual", "complexity": "simple", "requires_retrieval": True, "requires_calculation": False, "requires_data_analysis": False,
            "requires_comparison": False, "subtasks": [{"id": "t1", "tool": "search_knowledge_base", "goal": "def", "query": "overfitting"}]}
    p, src, adj, note = Planner(FakeLLM(plan=good)).plan(q, cat, ALL, None, f, 6)
    assert src == "llm" and tools(p) == ["search_knowledge_base"]
    p, src, adj, note = Planner(FakeLLM(fail_plan=True)).plan(q, cat, ALL, None, f, 6)
    assert src == "rules_fallback" and "failed" in note and tools(p)            # invalid LLM output is never treated as success
    with pytest.raises(Exception):
        Planner(FakeLLM(fail_plan=True), allow_fallback=False).plan(q, cat, ALL, None, f, 6)
    p, src, *_ = Planner(FakeLLM(), mode="rules").plan(q, cat, ALL, None, f, 6)
    assert src == "rules"


def test_typed_plan_validation():
    with pytest.raises(Exception):
        QuestionPlan(question_type="magic", complexity="simple", requires_retrieval=True, requires_calculation=False, requires_data_analysis=False, requires_comparison=False,
                     subtasks=[Subtask(id="t1", tool="x", goal="g")])
    with pytest.raises(Exception):
        Subtask(id="bad id", tool="x", goal="g")
