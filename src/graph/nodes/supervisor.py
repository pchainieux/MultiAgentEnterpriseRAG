from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.graph.state import GraphState

SupervisorDecision = Literal["plan_and_retrieve", "answer_directly", "clarify", "refuse"]

def _last_user_message(messages: list[BaseMessage]) -> str:
    """
    Extract the most recent user message from the LangChain message history to support supervisor routing heuristics.
    Inputs: messages conversation history; Outputs: the last HumanMessage content.
    """
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""

def decide_next_step(state: GraphState) -> SupervisorDecision:
    """
    Choose the next workflow branch (RAG, direct answer, or clarification). 
    Inputs: state containing at least 'question' and 'messages'; Outputs: a SupervisorDecision string ('plan_and_retrieve' | 'answer_directly' | 'clarify' | 'refuse').
    """
    question = (state.get("question") or "").strip()
    messages = state.get("messages") or []
    last_user = _last_user_message(messages)

    if not question:
        return "clarify"

    smalltalk = re.match(r"^(hi|hello|hey|thanks|thank you)\b", question.lower())
    meta = "how does this work" in question.lower() or "what can you do" in question.lower()
    if smalltalk or meta:
        return "answer_directly"

    referential = any(w in question.lower().split() for w in ["it", "this", "that", "they", "them"])
    if referential and len(last_user) < 5:
        return "clarify"

    return "plan_and_retrieve"

def supervisor_node(state: GraphState) -> dict:
    """
    Run the supervisor decision logic, store the chosen branch label in state, and append an internal trace message for debugging/observability.
    Inputs: state ; Outputs: a partial state update dict with 'supervisor_decision' and an appended 'messages' trace entry.
    """
    decision = decide_next_step(state)

    msg = AIMessage(
        content=f"Supervisor decision: {decision}",
        name="supervisor",
    )

    return {
        "messages": [msg],
        "supervisor_decision": decision,
    }
