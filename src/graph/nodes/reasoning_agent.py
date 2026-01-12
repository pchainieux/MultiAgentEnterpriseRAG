from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document

from src.graph.state import GraphState
from src.rag.llm.models import get_reasoning_llm
from src.rag.llm.prompts import REASONING_SYSTEM_PROMPT


def _format_docs_for_prompt(documents: List[Document]) -> str:
    """
    Convert a list of retrieved Document objects into a compact numbered context block for prompting the reasoning LLM.
    Inputs: documents retrieved chunks with metadata; Outputs: a single formatted string capped to a small number of docs an=d characters per doc.
    """
    lines: List[str] = []
    for i, d in enumerate(documents[:10]): 
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", meta.get("section", ""))
        lines.append(
            f"[{i}] source={src} page={page}\n{d.page_content[:1000]}"
        )
    return "\n\n".join(lines)


def reasoning_node(state: GraphState) -> dict:
    """
    Generate a draft answer to the current question using only the retrieved document context and a short window of recent chat history.
    Inputs: state ; Outputs: a partial state update dict with 'answer' and an assistant message trace.
    """
    llm = get_reasoning_llm()

    question = state.get("question", "")
    documents: List[Document] = state.get("documents", [])
    history: List = state.get("messages", [])

    context_block = _format_docs_for_prompt(documents)

    prompt_messages = [
        HumanMessage(content=REASONING_SYSTEM_PROMPT, name="system"),
    ]
    prompt_messages.extend(history[-4:])
    prompt_messages.append(
        HumanMessage(
            content=(
                "Using ONLY the context below, answer the user question.\n\n"
                f"Context:\n{context_block}\n\n"
                f"Question: {question}"
            ),
            name="user",
        )
    )

    resp = llm.invoke(prompt_messages)
    answer_text = resp.content

    assistant_msg = AIMessage(
        content=answer_text,
        name="reasoning_agent",
    )

    return {
        "answer": answer_text,
        "messages": [assistant_msg],
    }
