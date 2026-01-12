from __future__ import annotations

from typing import List, Optional, TypedDict, Any
from typing_extensions import Annotated

from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph import add_messages


class GraphState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    Each agent reads from this structure and returns a partial update that LangGraph will merge into the global state.
    """
    # Full conversational history
    messages: Annotated[List[BaseMessage], add_messages]

    # Current user question for this turn 
    question: str

    # Optional structured plan produced by QueryPlannerAgent
    plan: Optional[str]

    # Retrieved documents for this turn
    documents: List[Document]

    # Final answer string produced by ReasoningAgent
    answer: Optional[str]

    # Structured citation data 
    citations: List[dict[str, Any]]

    # Session identifier for multi‑turn / Redis memory
    session_id: Optional[str]

    # Small integer used for retries 
    retry_count: int

    # Supervisor routing decision for this turn
    supervisor_decision: Optional[str]

    # Used by quality gate to avoid infinite loops
    retrieval_attempted: bool
