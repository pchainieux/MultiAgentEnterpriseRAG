from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.app.schemas.chat import ChatRequest, ChatResponse
from src.graph.workflow import get_graph_app
from src.graph.state import GraphState

router = APIRouter(prefix="/chat", tags=["chat"])

def _convert_client_messages(client_messages):
    """
    Convert the client provided chat messages into LangChain BaseMessage objects for execution inside the LangGraph workflow.
    Inputs: client_messages ; Outputs: list of LangChain messages.
    """
    converted = []
    for m in client_messages:
        role = m.role
        if role == "user":
            converted.append(HumanMessage(content=m.content))
        elif role == "assistant":
            converted.append(AIMessage(content=m.content))
        elif role == "system":
            converted.append(SystemMessage(content=m.content))
        else:
            converted.append(HumanMessage(content=m.content))
    return converted


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Execute one chat turn by building an initial GraphState, invoking the compiled LangGraph workflow for the given session, and returning the final answer with citations.
    Inputs: payload containing session_id and messages ; Outputs: ChatResponse containing session_id, answer text, and structured citations.
    """
    try:
        graph_app = get_graph_app()

        session_id = (payload.session_id or "").strip()
        if not session_id:
            session_id = f"session-{uuid.uuid4()}"

        if not payload.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="messages cannot be empty",
            )
        last_user = next(
            (m for m in reversed(payload.messages) if m.role == "user"),
            None,
        )
        if last_user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="at least one user message is required",
            )

        question = last_user.content

        langchain_messages = _convert_client_messages(payload.messages)

        initial_state: GraphState = {
            "messages": langchain_messages,
            "question": question,
            "plan": None,
            "documents": [],
            "answer": None,
            "citations": [],
            "session_id": session_id,
            "retry_count": 0,
        }

        config = {"configurable": {"thread_id": session_id}}

        result_state: GraphState = await graph_app.ainvoke(
            initial_state,
            config=config,
        )

        answer = result_state.get("answer") or ""
        citations = result_state.get("citations", [])

        if not answer:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Graph returned empty answer",
            )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            citations=citations,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in chat workflow: {exc}",
        )
