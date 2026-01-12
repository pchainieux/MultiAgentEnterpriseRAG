from __future__ import annotations

from langchain_core.messages import AIMessage

from src.graph.state import GraphState


def clarify_node(state: GraphState) -> dict:
    """
    Ask the user for missing context when the query is empty or ambiguous.
    Inputs: state ; Outputs: a partial state update dict with a clarification 'answer', an assistant message, and empty 'citations'.
    """
    question = (state.get("question") or "").strip()

    if not question:
        clarification = "What would you like to know? Please ask a question about the documents you ingested."
    else:
        clarification = (
            "Could you clarify what you mean, and what document/topic you want me to use? "
            "For example: which file name and which section/page."
        )

    return {
        "answer": clarification,
        "messages": [AIMessage(content=clarification, name="clarify_agent")],
        "citations": [],
    }
