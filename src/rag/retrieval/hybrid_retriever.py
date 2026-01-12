from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.documents import Document
from qdrant_client import models

from src.app.core.config import settings
from src.rag.llm.models import get_embedding_model
from src.rag.retrieval.reranker import simple_rerank
from src.rag.vectorstore.qdrant_client import get_qdrant_client


@dataclass
class HybridRetriever:
    """
    Hybrid retriever: 
    - Dense: Qdrant vector search (query embedding), 
    - Lexical: Qdrant MatchText over payload["text"], 
    - Merge + rerank.
    """
    collection_name: str
    top_k: int = settings.RETRIEVAL_TOP_K
    dense_k: int = settings.RETRIEVAL_DENSE_K
    lexical_k: int = settings.RETRIEVAL_LEXICAL_K

    @classmethod
    def from_env(cls) -> "HybridRetriever":
        """
        Construct a HybridRetriever configured from environment settings.
        Inputs: cls ; Outputs: a HybridRetriever instance.
        """
        return cls(collection_name=settings.QDRANT_COLLECTION_NAME)

    def _dense_search(self, query: str, filters: Optional[models.Filter]) -> List[Document]:
        """
        Perform dense retrieval by embedding the query and running a Qdrant vector search, converting hits into LangChain Document objects with payload metadata.
        Inputs: search text, filters ; Outputs: a list[Document] of dense retrieved chunks.
        """
        client = get_qdrant_client()
        embedder = get_embedding_model()

        qvec = embedder.embed_documents([query])[0]
        hits = client.search(
            collection_name=self.collection_name,
            query_vector=qvec,
            query_filter=filters,
            limit=self.dense_k,
            with_vectors=False,
        )

        docs: List[Document] = []
        for h in hits:
            payload = h.payload or {}
            docs.append(Document(page_content=payload.get("text", ""), metadata=payload))
        return docs

    def _lexical_search(self, query: str, filters: Optional[models.Filter]) -> List[Document]:
        """
        Perform lexical retrieval by scrolling Qdrant points matching the query text against the stored payload field 'text'.
        Inputs: search text, filters ; Outputs: a list[Document] of lexically matched chunks.
        """
        client = get_qdrant_client()

        must = []
        if filters and getattr(filters, "must", None):
            must.extend(filters.must)

        must.append(
            models.FieldCondition(
                key="text",
                match=models.MatchText(text=query),
            )
        )

        lexical_filter = models.Filter(must=must)

        points, _ = client.scroll(
            collection_name=self.collection_name,
            scroll_filter=lexical_filter,
            with_vectors=False,
            limit=self.lexical_k,
        )

        docs: List[Document] = []
        for p in points:
            payload = p.payload or {}
            docs.append(Document(page_content=payload.get("text", ""), metadata=payload))
        return docs

    def retrieve(self, query: str, filters: Optional[models.Filter] = None) -> List[Document]:
        """
        Retrieve candidate chunks using dense and lexical search, merge duplicates using chunk identifiers, rerank candidates, and return the top results.
        Inputs: search text, filters ; Outputs: a list[Document] reranked and truncated to top_k.
        """
        dense = self._dense_search(query, filters)
        lexical = self._lexical_search(query, filters)

        merged: Dict[str, Document] = {}
        for d in (dense + lexical):
            meta = d.metadata or {}
            key = (
                str(meta.get("chunk_uid"))
                or f"{meta.get('source')}|{meta.get('page')}|{meta.get('chunk_id')}|{d.page_content[:50]}"
            )
            merged[key] = d

        reranked = simple_rerank(list(merged.values()), query=query, top_k=self.top_k)
        return reranked
