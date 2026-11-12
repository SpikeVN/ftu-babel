"""Document retrieval routes for Babel (single and batch fetching)."""

from __future__ import annotations

from typing import Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse

from server.models.schemas import DocumentBatchRequest, DocumentDetailResponse
from server.models.text import format_documents_as_text
from server.services.stash import Stash, get_stash

router = APIRouter(tags=["Documents"])


@router.post("/docs/fetch", response_model=list[DocumentDetailResponse])
@router.post("/api/fetch", response_model=list[DocumentDetailResponse])
@router.post("/docs", response_model=list[DocumentDetailResponse])
@router.post("/fetch", response_model=list[DocumentDetailResponse])
def batch_fetch_documents_post(
    payload: DocumentBatchRequest,
    request: Request,
    stash_instance: Stash = Depends(get_stash),
):
    """
    POST: Batch fetch full content and metadata for multiple documents in a single request.
    Accepts either {"doc_ids": [1525, 1526]} or {"ids": [1525, 1526]}.
    """
    clean_ids = payload.resolved_ids()
    if not clean_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide at least one valid document ID in 'doc_ids' or 'ids'.",
        )

    docs = stash_instance.get_documents_detail(doc_ids=clean_ids, clean=payload.clean)

    accept_header = request.headers.get("accept", "")
    wants_text = (payload.format == "text") or (
        payload.format is None and any(h in accept_header for h in ("text/plain", "text/markdown"))
    )

    if wants_text:
        body = format_documents_as_text(docs)
        return PlainTextResponse(content=body, media_type="text/plain; charset=utf-8")

    return docs


@router.get("/docs/batch", response_model=list[DocumentDetailResponse])
@router.get("/docs/fetch", response_model=list[DocumentDetailResponse])
@router.get("/api/docs", response_model=list[DocumentDetailResponse])
@router.get("/api/fetch", response_model=list[DocumentDetailResponse])
def batch_fetch_documents_get(
    request: Request,
    ids: str = Query(..., description="Comma-separated document IDs, e.g. '1525,1526'"),
    clean: bool = Query(True, description="Whether to strip links, images, and headers so only normal text remains"),
    format: Optional[Literal["json", "text"]] = Query(None, description="Response format: 'json' (default) or 'text' for RAG LLM context"),
    stash_instance: Stash = Depends(get_stash),
):
    """
    GET: Batch fetch full content and metadata for multiple documents in a single request via comma-separated IDs.
    """
    parsed_ids = [int(p.strip()) for p in ids.split(",") if p.strip().isdigit()]
    if not parsed_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid numeric document IDs provided in 'ids'.",
        )

    docs = stash_instance.get_documents_detail(doc_ids=parsed_ids, clean=clean)

    accept_header = request.headers.get("accept", "")
    wants_text = (format == "text") or (
        format is None and any(h in accept_header for h in ("text/plain", "text/markdown"))
    )

    if wants_text:
        body = format_documents_as_text(docs)
        return PlainTextResponse(content=body, media_type="text/plain; charset=utf-8")

    return docs


@router.get("/docs/{doc_id}", response_model=DocumentDetailResponse)
@router.get("/docs/{doc_id}/", response_model=DocumentDetailResponse)
def get_document_endpoint(
    doc_id: int,
    request: Request,
    format: Optional[Literal["json", "text"]] = Query(None, description="Response format: 'json' (default) or 'text' for raw LLM ingestion"),
    clean: bool = Query(True, description="Whether to strip links, images, and headers so only normal text remains"),
    stash_instance: Stash = Depends(get_stash),
):
    """Fetch full content and metadata for a single crawled document by doc_id."""
    doc = stash_instance.get_document_detail(doc_id=doc_id, clean=clean)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with doc_id {doc_id} not found.",
        )

    accept_header = request.headers.get("accept", "")
    wants_text = (format == "text") or (
        format is None and any(h in accept_header for h in ("text/plain", "text/markdown"))
    )

    if wants_text:
        header_lines = [
            f"Title: {doc['title']}",
            f"URL: {doc['url']}",
        ]
        if doc.get("date"):
            header_lines.append(f"Date: {doc['date']}")
        if doc.get("summary"):
            header_lines.append(f"Summary: {doc['summary']}")
        header_text = "\n".join(header_lines)
        body = f"{header_text}\n\n{doc['content']}"
        return PlainTextResponse(content=body, media_type="text/plain; charset=utf-8")

    return DocumentDetailResponse(**doc)
