from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.app.core.config import settings


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """
    Build the RecursiveCharacterTextSplitter configured from settings to produce overlapping text chunks suitable for embedding and storage.
    Inputs: none ; Outputs: a configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split loaded Documents into smaller overlapping chunks while preserving and augmenting metadata required for stable chunk identification.
    Inputs: loaded docs ; Outputs: chunks with metadata.
    """
    splitter = _get_splitter()
    chunks = splitter.split_documents(documents)

    for i, c in enumerate(chunks):
        meta = c.metadata or {}
        meta.setdefault("chunk_id", i)
        c.metadata = meta

    return chunks
