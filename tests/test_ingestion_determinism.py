# This test aimes to check that re-ingesting identical docs produces identical Qdrant point IDs

import pytest
from langchain_core.documents import Document

from src.rag.ingestion import indexing


class DummyQdrantClient:
    def __init__(self):
        self.upserts = []

    def upsert(self, collection_name, points, wait=True):
        self.upserts.append((collection_name, points, wait))


class DummyEmbedder:
    def __init__(self, dim=4):
        self.dim = dim

    def embed_documents(self, texts):
        return [[0.0] * self.dim for _ in texts]


def test_index_documents_chunk_uid_is_deterministic(monkeypatch):
    dummy_client = DummyQdrantClient()

    def fake_load_any(path):
        return [
            Document(
                page_content="Hello world. This is stable content.",
                metadata={"source": str(path), "page": 1, "type": "txt"},
            )
        ]

    def fake_chunk_documents(docs):
        return [
            Document(page_content="Hello world.", metadata={**docs[0].metadata, "chunk_id": 0}),
            Document(page_content="This is stable content.", metadata={**docs[0].metadata, "chunk_id": 1}),
        ]

    monkeypatch.setattr(indexing, "get_qdrant_client", lambda: dummy_client)
    monkeypatch.setattr(indexing, "ensure_collection", lambda: None)
    monkeypatch.setattr(indexing, "load_any", fake_load_any)
    monkeypatch.setattr(indexing, "chunk_documents", fake_chunk_documents)
    monkeypatch.setattr(indexing, "get_embedding_model", lambda: DummyEmbedder(dim=4))
    monkeypatch.setattr(indexing.settings, "QDRANT_VECTOR_DIM", 4)

    res1 = indexing.index_documents(["/tmp/a.txt"])
    res2 = indexing.index_documents(["/tmp/a.txt"])

    assert len(dummy_client.upserts) == 2
    points1 = dummy_client.upserts[0][1]
    points2 = dummy_client.upserts[1][1]

    assert res1.points_upserted == 2
    assert res2.points_upserted == 2

    ids1 = [p.id for p in points1]
    ids2 = [p.id for p in points2]
    assert ids1 == ids2
