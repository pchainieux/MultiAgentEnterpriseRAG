from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_txt(path: str | Path) -> List[Document]:
    """
    Load a .txt or .md file into a single LangChain Document with standardised source metadata.
    Inputs: path to file ; Outputs: a list containing one Document with page_content and metadata.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata={
                "source": str(p),
                "source_name": p.name,
                "type": "txt",
                "page": 1,
            },
        )
    ]


def load_pdf(path: str | Path) -> List[Document]:
    """
    Load a PDF into per-page LangChain Document objects and normalise metadata fields needed later for retrieval and citations.
    Inputs: path to file ; Outputs: a list[Document] (one per page) with metadata.
    """
    p = Path(path)
    loader = PyPDFLoader(str(p))
    docs = loader.load()

    for i, d in enumerate(docs):
        d.metadata.setdefault("source", str(p))
        d.metadata.setdefault("source_name", p.name)
        d.metadata.setdefault("page", d.metadata.get("page", i + 1))
        d.metadata.setdefault("type", "pdf")

    return docs


def load_any(path: str | Path) -> List[Document]:
    """
    Dispatch document loading based on file extension and return LangChain Document objects for supported formats.
    Inputs: path to file ; Outputs: list[Document] loaded from the file; raises ValueError for unsupported extensions.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in {".txt", ".md"}:
        return load_txt(p)
    if suffix == ".pdf":
        return load_pdf(p)

    raise ValueError(f"Unsupported file type: {suffix}")
