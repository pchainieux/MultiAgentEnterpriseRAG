# This test checks that the citation_node() refuses when there are no documents, when the LLM returns invalid JSON twice, or when citations are empty

import pytest
from langchain_core.documents import Document

from src.graph.nodes.citation_agent import citation_node


class DummyLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.invocations = 0

    def invoke(self, messages):
        self.invocations += 1
        content = self._responses.pop(0)
        return type("Resp", (), {"content": content})


def test_citation_node_refuses_when_no_documents():
    state = {"answer": "Some answer", "documents": []}
    out = citation_node(state)
    assert out["citations"] == []
    assert "don't know" in out["answer"].lower()


def test_citation_node_refuses_after_two_invalid_json(monkeypatch):
    dummy_llm = DummyLLM(["not json", "still not json"])

    monkeypatch.setattr(
        "src.graph.nodes.citation_agent.get_citation_llm",
        lambda: dummy_llm,
    )

    docs = [Document(page_content="evidence", metadata={"source": "a.txt", "page": 1, "doc_id": "d1"})]
    state = {"answer": "Claim", "documents": docs}

    out = citation_node(state)
    assert out["citations"] == []
    assert "can't provide" in out["answer"].lower()
    assert dummy_llm.invocations == 2


def test_citation_node_refuses_when_citations_empty(monkeypatch):
    dummy_llm = DummyLLM(['{"answer_with_citations":"Claim [0]","citations": []}'])

    monkeypatch.setattr(
        "src.graph.nodes.citation_agent.get_citation_llm",
        lambda: dummy_llm,
    )

    docs = [Document(page_content="evidence", metadata={"source": "a.txt", "page": 1, "doc_id": "d1"})]
    state = {"answer": "Claim", "documents": docs}

    out = citation_node(state)
    assert out["citations"] == []
    assert "can't provide" in out["answer"].lower()

