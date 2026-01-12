from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from qdrant_client import models

from src.app.core.config import settings
from src.rag.vectorstore.qdrant_client import get_qdrant_client, ensure_collection
from src.rag.ingestion.loaders import load_any
from src.rag.ingestion.chunking import chunk_documents
from src.rag.llm.models import get_embedding_model


@dataclass(frozen=True)
class IngestionResult:
    indexed_files: List[str]
    documents_loaded: int
    chunks_indexed: int
    points_upserted: int

_CHUNK_UID_NAMESPACE = uuid.UUID("2b1c9b44-8bfe-4b18-9a3f-5db2b9de6e8a")

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _embed_chunks(chunks: List[Document]) -> List[List[float]]:
    embedder = get_embedding_model()
    texts = [c.page_content for c in chunks]
    return embedder.embed_documents(texts)


def index_documents(paths: Iterable[str]) -> IngestionResult:
    """
    End-to-end ingestion entrypoint: load documents from paths, chunk them, embed chunks, ensure the Qdrant collection exists, and upsert deterministic chunk points.
    Inputs: paths to ingest; Outputs: IngestionResult containing counts and the list of indexed file paths.
    """
    client = get_qdrant_client()

    indexed_files: List[str] = []
    documents_loaded = 0
    all_chunks: List[Document] = []

    for path in paths:
        docs = load_any(path)
        indexed_files.append(str(path))
        documents_loaded += len(docs)

        chunks = chunk_documents(docs)
        all_chunks.extend(chunks)

    if not all_chunks:
        return IngestionResult(
            indexed_files=indexed_files,
            documents_loaded=documents_loaded,
            chunks_indexed=0,
            points_upserted=0,
        )

    vectors = _embed_chunks(all_chunks)

    dim = len(vectors[0]) if vectors else None
    if dim and dim != settings.QDRANT_VECTOR_DIM:
        raise ValueError(
            f"Embedding dim ({dim}) != QDRANT_VECTOR_DIM ({settings.QDRANT_VECTOR_DIM}). "
            "Fix settings.QDRANT_VECTOR_DIM or change the embedding model."
        )

    ensure_collection()

    points: List[models.PointStruct] = []
    for doc, vec in zip(all_chunks, vectors):
        meta = doc.metadata or {}

        source = str(meta.get("source", "unknown"))
        page = meta.get("page", meta.get("section", None))
        chunk_id = meta.get("chunk_id", None)

        doc_id = str(meta.get("doc_id") or _sha1(source))

        content_prefix = (doc.page_content or "")[:120]
        stable_key = f"{doc_id}|{page}|{chunk_id}|{content_prefix}"
        chunk_uuid = str(uuid.uuid5(_CHUNK_UID_NAMESPACE, stable_key))

        meta.setdefault("source", source)
        meta.setdefault("source_name", Path(source).name if source != "unknown" else "unknown")
        meta.setdefault("doc_id", doc_id)
        meta.setdefault("id", doc_id)  
        meta.setdefault("chunk_uid", chunk_uuid) 
        meta.setdefault("chunk_id", chunk_id)
        meta.setdefault("page", page)
        meta.setdefault("type", meta.get("type", "unknown"))

        points.append(
            models.PointStruct(
                id=chunk_uuid,
                vector=vec,
                payload={
                    "text": doc.page_content,
                    **meta,
                },
            )
        )

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return IngestionResult(
        indexed_files=indexed_files,
        documents_loaded=documents_loaded,
        chunks_indexed=len(all_chunks),
        points_upserted=len(points),
    )
