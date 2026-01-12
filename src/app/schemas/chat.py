from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant | system")
    content: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Stable conversation id; if omitted, backend can generate one.",
    )
    messages: List[ChatMessage] = Field(
        ..., description="Full chat history for this request (client-side view)."
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[dict]
