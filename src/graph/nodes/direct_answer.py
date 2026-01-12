from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.graph.state import GraphState
from src.rag.llm.models import get_reasoning_llm
from src.rag.llm.prompts import DIRECT_SYSTEM_PROMPT

def direct_answer_node(state: GraphState) -> dict:
    """
    Answer "small talk" questions directly using the LLM without performing document retrieval, and explicitly return no citations.
    Inputs: state ; Outputs: a partial state update dict with 'answer', an assistant message, and 'citations' set to an empty list.
    """
    llm = get_reasoning_llm()
    question = (state.get("question") or "").strip()

    prompt_messages = [
        SystemMessage(content=DIRECT_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    resp = llm.invoke(prompt_messages)
    answer = (resp.content or "").strip()

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer, name="direct_answer")],
        "citations": [],
    }
