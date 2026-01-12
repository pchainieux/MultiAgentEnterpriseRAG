from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.graph.state import GraphState
from src.rag.retrieval.hybrid_retriever import HybridRetriever

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """
    Initialise and cache a HybridRetriever so retrieval configuration and underlying clients are reused across requests.
    Inputs: none ; Outputs: a configured HybridRetriever instance.
    """
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever.from_env()
    return _retriever


def retrieval_node(state: GraphState) -> dict:
    """
    Retrieve candidate document chunks for the current question and attach them to the graph state.
    Inputs: state ; Outputs: a partial state update dict with 'documents' and a retrieval summary message.
    """
    retriever = _get_retriever()

    question = state.get("question", "")
    plan = state.get("plan", None)

    query_text = question if not plan else f"{question}\n\nPlan:\n{plan}"

    docs: List[Document] = retriever.retrieve(query_text)

    retrieval_summary = AIMessage(
        content=f"Retrieved {len(docs)} documents for the query.",
        name="retrieval_agent",
    )

    return {
        "documents": docs,
        "messages": [retrieval_summary],
    }
