from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.app.core.config import settings
from src.graph.state import GraphState
from src.rag.memory.redis_memory import (
    load_memory_bundle_from_redis,
    save_memory_bundle_to_redis,
)

INTERNAL_AGENT_NAMES = {
    "supervisor",
    "query_planner",
    "retrieval_agent",
    "reasoning_agent",
    "citation_agent",
    "memory_agent",
}


def _is_user_visible_message(m: BaseMessage) -> bool:
    """
    Decide whether a message should be persisted to Redis by filtering out internal agent trace AI messages.
    Inputs: BaseMessage from the graph message history; Outputs: True if the message is user visible and should be stored, otherwise False.
    """
    if isinstance(m, SystemMessage):
        return True

    if isinstance(m, HumanMessage):
        return True

    if isinstance(m, AIMessage):
        if getattr(m, "name", None) in INTERNAL_AGENT_NAMES:
            return False
        return True

    return False


def _truncate(s: str, max_chars: int) -> str:
    """
    Truncate a string to a maximum character length for safe storage in Redis summaries and debug friendly logs.
    Inputs: s (str) raw text and max_chars (int); Outputs: a stripped string truncated to max_chars.
    """
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[: max_chars - 3] + "..."


def _update_summary(existing_summary: str, overflow: List[BaseMessage]) -> str:
    """
    Update the rolling conversation summary deterministically (no LLM) by appending short bullet lines derived from messages that overflow the retention window.
    Inputs: existing_summary and overflow messages being dropped from the window; Outputs: an updated summary string.
    """
    if not overflow:
        return existing_summary

    lines: List[str] = []
    for m in overflow:
        if isinstance(m, HumanMessage):
            lines.append(f"- User: {_truncate(m.content, 200)}")
        elif isinstance(m, AIMessage):
            lines.append(f"- Assistant: {_truncate(m.content, 200)}")

    delta = "\n".join(lines).strip()
    combined = (existing_summary.strip() + "\n" + delta).strip() if existing_summary else delta
    return _truncate(combined, settings.MEMORY_SUMMARY_MAX_CHARS)


def load_memory_node(state: GraphState) -> GraphState:
    """
    Load a session's summary and recent message window from Redis and prepend them to the current request messages.
    Inputs: state ; Outputs: a partial state update dict that replaces/prepends 'messages' with injected memory context.
    """
    session_id = state.get("session_id")
    existing = state.get("messages") or []

    if not session_id:
        return {"messages": existing}

    summary, recent = load_memory_bundle_from_redis(session_id)

    summary_msg: List[BaseMessage] = []
    if summary.strip():
        summary_msg = [
            SystemMessage(content=f"Conversation summary (for context):\n{summary}")
        ]

    if len(existing) > 1:
        return {"messages": summary_msg + existing}

    return {"messages": summary_msg + recent + existing}


def save_memory_node(state: GraphState) -> dict:
    """
    Persist a bounded window of user visible messages plus a rolling summary to Redis for multi turn continuity. 
    Inputs: state ; Outputs: a partial state update dict. 
    """
    session_id = state.get("session_id")
    messages: List[BaseMessage] = state.get("messages", [])
    final_answer = (state.get("answer") or "").strip()

    if not session_id:
        return {"messages": messages}

    visible = [m for m in messages if _is_user_visible_message(m)]

    if final_answer:
        while visible and isinstance(visible[-1], AIMessage):
            visible.pop()
        visible.append(AIMessage(content=final_answer))

    existing_summary, existing_recent = load_memory_bundle_from_redis(session_id)

    merged = (existing_recent + visible) if existing_recent else visible

    max_n = settings.MEMORY_MAX_MESSAGES
    if len(merged) > max_n:
        overflow = merged[: len(merged) - max_n]
        merged = merged[-max_n:]
        new_summary = _update_summary(existing_summary, overflow)
    else:
        new_summary = existing_summary

    save_memory_bundle_to_redis(
        session_id,
        summary=new_summary,
        messages=merged,
        ttl_seconds=settings.REDIS_TTL_SECONDS,
    )

    return {"messages": messages}
