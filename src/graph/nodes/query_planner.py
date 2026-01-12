from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import GraphState
from src.rag.llm.models import get_planner_llm
from src.rag.llm.prompts import QUERY_PLANNER_SYSTEM_PROMPT


def query_planner_node(state: GraphState) -> dict:
    """
    Produce a short retrieval plan from the current question and recent chat context to improve downstream retrieval.
    Inputs: state ; Outputs: a partial state update dict with 'plan' text and a planner message trace.
    """
    llm = get_planner_llm()

    original_question = state.get("question", "")
    history: List = state.get("messages", [])

    prompt_messages = [
        HumanMessage(content=QUERY_PLANNER_SYSTEM_PROMPT, name="system"),
    ]
    prompt_messages.extend(history[-4:]) 

    retry_count = int(state.get("retry_count", 0) or 0)
    if retry_count > 0:
        prompt_messages.append(
            HumanMessage(
                content=(
                    "Retry mode: rewrite the query to improve retrieval. "
                    "Use alternative keywords, expand acronyms, and add synonyms. "
                    "Keep it short and searchable."
                ),
                name="user",
            )
        )

    prompt_messages.append(
        HumanMessage(
            content=f"User question: {original_question}",
            name="user",
        )
    )

    planner_response = llm.invoke(prompt_messages)

    plan_text = planner_response.content

    planner_message = AIMessage(
        content=plan_text,
        name="query_planner",
    )

    return {
        "plan": plan_text,
        "messages": [planner_message],
    }

