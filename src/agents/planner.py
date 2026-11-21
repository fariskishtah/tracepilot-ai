"""LLM planner (schema-constrained JSON, Pydantic-validated) with guard + rule fallback."""
from __future__ import annotations

import json

from src.agents.routing import analyze_question, guard_plan, rule_plan
from src.agents.state import LLMPlan, QuestionPlan
from src.llm import LLMError
from src.security.guard import SYSTEM_SAFETY

TOOL_BRIEF = {
    "search_knowledge_base": "search the documents for facts/definitions/explanations (fields: query)",
    "analyze_dataset": "exact statistics on a CSV (fields: dataset_id, operation in {row_count,describe_columns,mean,median,min,max,sum,std,var,count,value_counts,groupby_agg,correlation,select_rows,lookup_value}, column, group_by, agg, filters[{column,op,value}] - use filters to pick ONE row for lookup_value, e.g. {column:'run_id',op:'==',value:'X1'} or {column:'name',op:'contains',value:'text'}; column must be one of the dataset's columns)",
    "calculate_expression": "arithmetic on earlier numeric results (fields: expression using earlier subtask ids as variables, e.g. 't2 - t1'; depends_on)",
    "compare_evidence": "compare retrieved sources (fields: aspects[], depends_on=search subtask ids)",
    "summarize_evidence": "summarize retrieved sources (fields: depends_on=search subtask ids)",
    "retrieve_source_context": "read neighbouring text of a retrieved source (depends_on)",
    "refine_search_query": "rewrite/decompose queries (usually not needed in a plan)",
}

EXAMPLES = """Example A (simple fact)
Question: What is a Python tuple?
{"question_type":"factual","complexity":"simple","requires_retrieval":true,"requires_calculation":false,"requires_data_analysis":false,"requires_comparison":false,"subtasks":[{"id":"t1","tool":"search_knowledge_base","goal":"find the definition of a tuple","query":"Python tuple definition immutable"}]}

Example B (comparison, two searches then compare)
Question: Compare tuples and sets in Python.
{"question_type":"comparison","complexity":"moderate","requires_retrieval":true,"requires_calculation":false,"requires_data_analysis":false,"requires_comparison":true,"subtasks":[{"id":"t1","tool":"search_knowledge_base","goal":"evidence on tuples","query":"Python tuples"},{"id":"t2","tool":"search_knowledge_base","goal":"evidence on sets","query":"Python sets"},{"id":"t3","tool":"compare_evidence","goal":"compare tuples and sets","depends_on":["t1","t2"],"aspects":["tuples","sets"]}]}

Example C (two table lookups then arithmetic)
Question: How much larger is the amount of order O005 than order O010 in the orders data?
{"question_type":"calculation","complexity":"moderate","requires_retrieval":false,"requires_calculation":true,"requires_data_analysis":true,"requires_comparison":false,"subtasks":[{"id":"t1","tool":"analyze_dataset","goal":"amount of order O005","dataset_id":"data_orders","operation":"lookup_value","column":"amount","filters":[{"column":"order_id","op":"==","value":"O005"}]},{"id":"t2","tool":"analyze_dataset","goal":"amount of order O010","dataset_id":"data_orders","operation":"lookup_value","column":"amount","filters":[{"column":"order_id","op":"==","value":"O010"}]},{"id":"t3","tool":"calculate_expression","goal":"difference of the two amounts","expression":"t1 - t2","depends_on":["t1","t2"]}]}

Example D (arithmetic on numbers written in the question: calculator only, no search, no dataset)
Question: What is 8% of 250 plus 4?
{"question_type":"calculation","complexity":"simple","requires_retrieval":false,"requires_calculation":true,"requires_data_analysis":false,"requires_comparison":false,"subtasks":[{"id":"t1","tool":"calculate_expression","goal":"compute 8 percent of 250 plus 4","expression":"percent_of(8, 250) + 4"}]}

Example E (dataset statistic then concept)
Question: What is the maximum amount in the orders data and what does a refund mean?
{"question_type":"multi_step","complexity":"moderate","requires_retrieval":true,"requires_calculation":false,"requires_data_analysis":true,"requires_comparison":false,"subtasks":[{"id":"t1","tool":"analyze_dataset","goal":"maximum amount","dataset_id":"data_orders","operation":"max","column":"amount"},{"id":"t2","tool":"search_knowledge_base","goal":"meaning of refund","query":"refund meaning"}]}"""


def build_prompt(question: str, catalog: list[dict], enabled: list[str], scope: list[str] | None) -> tuple[str, str]:
    tools = "\n".join(f"- {n}: {TOOL_BRIEF[n]}" for n in enabled if n in TOOL_BRIEF and n != "refine_search_query")
    ds = "\n".join(f"- {d['dataset_id']} (course {d['course_id']}, {d['n_rows']} rows): columns {d['columns']}" for d in catalog) or "- (none)"
    sys = (SYSTEM_SAFETY + " You are the PLANNER of an agent. Decide the minimal sequence of tool calls needed to answer the user's question. "
           "Use only the available tools. Do NOT add steps that the question does not need (a plain fact needs ONE search). "
           "Use analyze_dataset ONLY when the question refers to a CSV/dataset/table and needs a number from it; use calculate_expression with the literal numbers when the question itself contains the numbers, calculate_expression only to combine earlier numeric results. "
           "Decompose multi-part questions into separate searches. If the question asks to explain, define or say why, include a search_knowledge_base step for it even when numbers are also needed. Answer with JSON only.")
    user = (f"Available tools:\n{tools}\n\nDatasets (CSV):\n{ds}\n\nSelected courses: {scope or 'all'}\n\n{EXAMPLES}\n\n"
            f"Now plan for this question (max 6 subtasks, ids t1, t2, ...):\nQuestion: {question}")
    return sys, user


class Planner:
    def __init__(self, llm, mode: str = "llm", allow_fallback: bool = True):
        self.llm, self.mode, self.allow_fallback = llm, mode, allow_fallback

    def plan(self, question: str, catalog: list[dict], enabled: list[str], scope, features: dict, max_steps: int, value_lookup=None):
        """Returns (plan, source, adjustments, notes)."""
        notes = ""
        if self.mode == "rules":
            p = rule_plan(question, catalog, features, enabled, scope, value_lookup)
            p, adj = guard_plan(p, question, features, enabled, catalog, max_steps)
            return p, "rules", adj, notes
        try:
            sys, user = build_prompt(question, catalog, enabled, scope)
            try:
                lp = self.llm.structured(sys, user, LLMPlan)
            except LLMError:                       # one retry with a different sampling seed
                lp = self.llm.structured(sys, user, LLMPlan, retry_seed=True)
            p = QuestionPlan(**lp.model_dump(), rationale=f"LLM plan: {len(lp.subtasks)} step(s)")
            src = "llm"
        except LLMError as e:
            if not self.allow_fallback:
                raise
            notes = f"LLM planner failed ({e}); using rule-based plan"
            p, src = rule_plan(question, catalog, features, enabled, scope, value_lookup), "rules_fallback"
        p, adj = guard_plan(p, question, features, enabled, catalog, max_steps)
        return p, src, adj, notes
