from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """
    Request body for path based ingestion.

    Example:
      {
        "paths": ["./data/docs/document_1.pdf", "./data/docs/document_2.md"]
      }
    """
    paths: List[str] = Field(..., description="List of file paths to ingest.")

class IngestResponse(BaseModel):
    indexed_files: List[str]
    documents_loaded: int
    chunks_indexed: int
    points_upserted: int
