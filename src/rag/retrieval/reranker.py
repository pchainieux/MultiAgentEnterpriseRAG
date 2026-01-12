from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document


def simple_rerank(docs: List[Document], query: str, top_k: int = 8) -> List[Document]:
    """
    Deterministically rerank retrieved documents by scoring simple query term presence, providing a zero-external-call placeholder for a real reranker.
    Inputs: docs candidates, user/query text, top_k maximum results; Outputs: a list[Document] sorted by heuristic relevance, truncated to top_k.
    """
    if not docs:
        return []

    terms = {t.lower() for t in query.split() if t.strip()}

    def score(doc: Document) -> Tuple[int, int]:
        text = doc.page_content.lower()
        hits = sum(1 for t in terms if t in text)
        return (hits, len(text))

    ranked = sorted(docs, key=score, reverse=True)
    return ranked[:top_k]
