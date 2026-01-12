from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, START, END

from src.graph.state import GraphState
from src.graph.nodes.supervisor import supervisor_node
from src.graph.nodes.query_planner import query_planner_node
from src.graph.nodes.retrieval_agent import retrieval_node
from src.graph.nodes.retrieval_quality_gate import quality_gate_node
from src.graph.nodes.reasoning_agent import reasoning_node
from src.graph.nodes.citation_agent import citation_node
from src.graph.nodes.memory_agent import load_memory_node, save_memory_node
from src.graph.nodes.direct_answer import direct_answer_node
from src.graph.nodes.clarify_agent import clarify_node


def supervisor_router(state: GraphState,) -> Literal["plan_and_retrieve", "answer_directly", "clarify", "refuse"]:
    """
    Route execution after the Supervisor node by mapping `state["supervisor_decision"]` to a valid workflow branch.
    Inputs: state (GraphState) containing `supervisor_decision`; Outputs: a branch label string for LangGraph conditional edges.
    """
    decision = state.get("supervisor_decision", "plan_and_retrieve")
    if decision not in ("plan_and_retrieve", "answer_directly", "clarify", "refuse"):
        return "plan_and_retrieve"
    return decision


def quality_gate_router(state: GraphState) -> Literal["retry", "continue"]:
    """
    Route execution after the retrieval quality gate by deciding whether to retry planning/retrieval or continue to reasoning.
    Inputs: state (GraphState) containing `retry_count` and `documents`; Outputs: "retry" or "continue" for LangGraph conditional edges.
    """
    retry_count = int(state.get("retry_count", 0) or 0)
    docs = state.get("documents") or []
    if retry_count == 0 and len(docs) < 2:
        return "retry"
    return "continue"


def build_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph by registering all nodes and wiring the main branches, retry loop, and terminal edges.
    Inputs: none; Outputs: a compiled StateGraph workflow definition ready to be executed by the API layer.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("supervisor", supervisor_node)

    workflow.add_node("clarify", clarify_node)
    workflow.add_node("direct_answer", direct_answer_node)

    workflow.add_node("query_planner", query_planner_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("quality_gate", quality_gate_node)

    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("citation", citation_node)
    workflow.add_node("save_memory", save_memory_node)

    workflow.add_edge(START, "load_memory")
    workflow.add_edge("load_memory", "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "plan_and_retrieve": "query_planner",
            "answer_directly": "direct_answer",
            "clarify": "clarify",
            "refuse": "clarify", 
        },
    )

    workflow.add_edge("clarify", "save_memory")
    workflow.add_edge("direct_answer", "save_memory")

    workflow.add_edge("query_planner", "retrieval")
    workflow.add_edge("retrieval", "quality_gate")

    workflow.add_conditional_edges(
        "quality_gate",
        quality_gate_router,
        {
            "retry": "query_planner",
            "continue": "reasoning",
        },
    )

    workflow.add_edge("reasoning", "citation")
    workflow.add_edge("citation", "save_memory")
    workflow.add_edge("save_memory", END)

    return workflow


_graph_app = build_graph().compile()


def get_graph_app():
    """
    Return the compiled LangGraph application instance used by the `/chat` endpoint to run the multi agent workflow.
    Inputs: none; Outputs: a compiled LangGraph app object.
    """
    return _graph_app
