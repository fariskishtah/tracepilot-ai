from __future__ import annotations

CAP = {"search_knowledge_base": "search", "retrieve_source_context": "search", "analyze_dataset": "analyze", "calculate_expression": "calculate",
       "compare_evidence": "compare", "summarize_evidence": "summarize", "refine_search_query": "refine"}


def tool_scores(item: dict, res: dict) -> dict:
    seq = res["tool_sequence"]
    caps = [CAP[t] for t in seq]
    exec_set, req, forb = set(caps), set(item["requires_tools"]), set(item["forbidden_tools"])
    prec = len(exec_set & req) / len(exec_set) if exec_set else None
    rec = len(exec_set & req) / len(req) if req else None
    # a search RETRY after weak evidence is justified: count only the distinct (tool, step) pairs for 'unnecessary'
    unnecessary = sum(1 for c in caps if c not in req and c != "refine")
    return {"tool_precision": prec, "tool_recall": rec, "required_ok": req <= exec_set, "forbidden_used": bool(exec_set & forb),
            "correct_path": req <= exec_set and not (exec_set & forb), "unnecessary_calls": unnecessary, "n_calls": len(caps),
            "unnecessary_rate": unnecessary / len(caps) if caps else 0.0, "capabilities": sorted(exec_set)}


def plan_completed(res: dict) -> bool | None:
    st = (res.get("metrics") or {}).get("step_status")
    if not st:
        return None
    return all(v in ("done",) for v in st.values())
