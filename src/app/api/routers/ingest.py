from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from src.app.api.deps import common_dependencies
from src.app.schemas.ingest import IngestRequest, IngestResponse
from src.rag.ingestion.indexing import index_documents

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/", response_model=IngestResponse)
async def ingest_endpoint(
    payload: IngestRequest,
    deps=Depends(common_dependencies),
) -> IngestResponse:
    """
    Ingest documents into the vector store by indexing a list of container local file paths and returning ingestion statistics for debugging/demo purposes.
    Inputs: payload with 'paths' and deps ; Outputs: IngestResponse containing indexed_files and ingestion counts.
    """
    logger = deps["logger"]

    if not payload.paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="paths must not be empty",
        )

    try:
        logger.info("ingest_start", paths=payload.paths)
        index_documents(payload.paths)
        logger.info("ingest_success", paths=payload.paths)
        result = index_documents(payload.paths)

        return IngestResponse(
            indexed_files=result.indexed_files,
            documents_loaded=result.documents_loaded,
            chunks_indexed=result.chunks_indexed,
            points_upserted=result.points_upserted,
        )
    except Exception as exc:
        logger.error("ingest_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )



