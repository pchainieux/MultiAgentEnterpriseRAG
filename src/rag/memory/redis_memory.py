from __future__ import annotations

import json
from typing import List, Tuple

import redis
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from src.app.core.config import settings

_redis_client: redis.Redis | None = None


def _get_redis_client() -> redis.Redis:
    """
    Create and cache a Redis client using settings.REDIS_URL so memory operations share a single connection pool.
    Inputs: none; Outputs: a redis.Redis client instance.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def _messages_key(session_id: str) -> str:
    """
    Build the Redis key used to store the serialised recent message window for a given chat session.
    Inputs: session_id ; Outputs: a Redis key string.
    """
    return f"chat:{session_id}:messages"


def _summary_key(session_id: str) -> str:
    """
    Build the Redis key used to store the rolling summary for a given chat session.
    Inputs: session_id ; Outputs: a Redis key string.
    """
    return f"chat:{session_id}:summary"


def load_memory_bundle_from_redis(session_id: str) -> Tuple[str, List[BaseMessage]]:
    """
    Load the conversation memory bundle for a session from Redis, with backward compatible fallback to a legacy single key format.
    Inputs: session_id ; Outputs: (summary (str), recent_messages (list[BaseMessage])).
    """
    try:
        client = _get_redis_client()

        summary = client.get(_summary_key(session_id)) or ""

        raw_messages = client.get(_messages_key(session_id))
        if raw_messages:
            data = json.loads(raw_messages)
            return summary, messages_from_dict(data)

        legacy_raw = client.get(f"chat:{session_id}")
        if not legacy_raw:
            return summary, []

        legacy_data = json.loads(legacy_raw)
        return summary, messages_from_dict(legacy_data)

    except Exception:
        return "", []


def save_memory_bundle_to_redis(
    session_id: str,
    *,
    summary: str,
    messages: List[BaseMessage],
    ttl_seconds: int | None = None,
) -> None:
    """
    Persist the conversation memory bundle to Redis with a TTL to enable multi turn continuity across requests.
    Inputs: session_id, summary, messages, ttl_seconds ; Outputs: None.
    """
    try:
        client = _get_redis_client()
        ttl = ttl_seconds if ttl_seconds is not None else settings.REDIS_TTL_SECONDS

        client.set(_summary_key(session_id), summary, ex=ttl)
        client.set(_messages_key(session_id), json.dumps(messages_to_dict(messages)), ex=ttl)

    except Exception:
        return
