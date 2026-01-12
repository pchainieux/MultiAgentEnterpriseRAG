from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from src.graph.state import GraphState

GateDecision = Literal["retry", "continue"]

def _gate_decision(state: GraphState) -> GateDecision:
    """
    Decide whether retrieval results are strong enough to continue, or whether to trigger a single retry to improve recall.
    Inputs: state containing 'documents' and 'retry_count'; Outputs: 'retry' if retrieval is weak and retry budget remains, otherwise 'continue'.
    """
    docs = state.get("documents") or []
    retry_count = int(state.get("retry_count", 0) or 0)

    if retry_count >= 1:
        return "continue"

    if len(docs) == 0:
        return "retry"
    if len(docs) < 2:
        return "retry"

    return "continue"

def quality_gate_node(state: GraphState) -> dict:
    """
    Apply the retrieval quality decision by incrementing retry_count on retry and always appending a trace message used for debugging the retry loop.
    Inputs: state ; Outputs: a partial state update dict that may include an updated 'retry_count' and always includes 'messages'.
    """
    decision = _gate_decision(state)

    if decision == "retry":
        new_retry = int(state.get("retry_count", 0) or 0) + 1
        msg = AIMessage(
            content=f"Retrieval quality gate: retrying retrieval (retry_count={new_retry}).",
            name="quality_gate",
        )
        return {
            "retry_count": new_retry,
            "messages": [msg],
        }

    msg = AIMessage(content="Retrieval quality gate: continue.", name="quality_gate")
    return {"messages": [msg]}
