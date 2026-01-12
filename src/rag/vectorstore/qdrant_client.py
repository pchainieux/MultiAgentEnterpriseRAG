from __future__ import annotations

from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient, models

from src.app.core.config import settings


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Create and cache a singleton QdrantClient configured from environment settings for reuse across ingestion and retrieval calls.
    Inputs: none; Outputs: a connected QdrantClient instance.
    """
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
        prefer_grpc=False,
    )


def ensure_collection(
    collection_name: Optional[str] = None,
    vector_size: Optional[int] = None,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    """
    Ensure the configured Qdrant collection exists by creating it if missing, using the provided vector size and distance metric.
    Inputs: collection_name, vector_size, distance; Outputs: None.
    """
    client = get_qdrant_client()
    name = collection_name or settings.QDRANT_COLLECTION_NAME
    dim = vector_size or settings.QDRANT_VECTOR_DIM

    collections = client.get_collections().collections
    if any(c.name == name for c in collections):
        return

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=dim,
            distance=distance,
        ),
    )
