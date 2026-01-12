# This test checks that HybridRetriever.retrieve() deduplicates candidates correctly (prefer chunk_uid when present; otherwise fall back to the composite key) and then rerank/truncate

from langchain_core.documents import Document
from src.rag.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_dedup_by_chunk_uid_and_fallback_key(monkeypatch):
    """
    Dense + lexical candidates should merge/dedup correctly:
    - If chunk_uid exists, it is the dedup key.
    - If chunk_uid is missing/empty, fallback composite key is used.
    """
    r = HybridRetriever(collection_name="documents", top_k=10, dense_k=10, lexical_k=10)

    dense_docs = [
        Document(page_content="alpha beta", metadata={"chunk_uid": "u1", "source": "s", "page": 1, "chunk_id": 0}),
        Document(page_content="gamma delta", metadata={"source": "s", "page": 2, "chunk_id": 1}),
    ]
    lexical_docs = [
        Document(page_content="alpha beta (duplicate)", metadata={"chunk_uid": "u1"}),  # duplicate by chunk_uid
        Document(page_content="gamma delta", metadata={"source": "s", "page": 2, "chunk_id": 1}), 
        Document(page_content="epsilon", metadata={"chunk_uid": "u2"}),
    ]

    monkeypatch.setattr(HybridRetriever, "_dense_search", lambda self, q, f: dense_docs)
    monkeypatch.setattr(HybridRetriever, "_lexical_search", lambda self, q, f: lexical_docs)

    out = r.retrieve("alpha")
    assert len(out) == 3
    assert sum(1 for d in out if (d.metadata or {}).get("chunk_uid") == "u1") == 1
