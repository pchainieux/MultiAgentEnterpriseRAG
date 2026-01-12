# This test checks that load_memory_bundle_from_redis() falls back to the chat:{session_id} key when new keys are missing.

import json
from langchain_core.messages import HumanMessage

from src.rag.memory import redis_memory


class DummyRedis:
    def __init__(self, store):
        self.store = dict(store)

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value


def test_load_memory_bundle_falls_back_to_legacy_key(monkeypatch):
    session_id = "abc"
    legacy_key = f"chat:{session_id}"

    legacy_messages = [{"type": "human", "data": {"content": "hi"}}]
    dummy = DummyRedis({legacy_key: json.dumps(legacy_messages)})

    monkeypatch.setattr(redis_memory, "_get_redis_client", lambda: dummy)

    summary, msgs = redis_memory.load_memory_bundle_from_redis(session_id)
    assert summary == ""  
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "hi"
