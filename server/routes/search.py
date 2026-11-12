"""Search routes for Babel."""

from __future__ import annotations

from typing import Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse

from server.models.schemas import SearchRequest, SingleSearchResult
from server.models.text import format_search_results_as_text
from server.services.stash import Stash, get_stash

router = APIRouter(tags=["Search"])


@router.get("/search", response_model=list[SingleSearchResult])
@router.get("/api/search", response_model=list[SingleSearchResult])
def search_endpoint_get(
    request: Request,
    q: str = Query(..., description="Search query string"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results to return"),
    hybrid_blend: float = Query(0.5, ge=0.0, le=1.0, description="Keyword vs vector blend (0.0 = vector only, 1.0 = keyword only)"),
    format: Optional[Literal["json", "text"]] = Query(None, description="Response format: 'json' (default) or 'text' for RAG LLM context"),
    stash_instance: Stash = Depends(get_stash),
):
    """GET: Hybrid semantic vector and keyword search across indexed documents with zerank-2 reranking."""
    if not q.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty.",
        )
    results = stash_instance.search_hybrid(
        query_text=q,
        top_k=limit,
        hybrid_blend=hybrid_blend,
        as_search_results=True,
    )

    accept_header = request.headers.get("accept", "")
    wants_text = (format == "text") or (format is None and any(h in accept_header for h in ("text/plain", "text/markdown")))

    if wants_text:
        body = format_search_results_as_text(results)
        return PlainTextResponse(content=body, media_type="text/plain; charset=utf-8")

    return results


@router.post("/search", response_model=list[SingleSearchResult])
@router.post("/api/search", response_model=list[SingleSearchResult])
def search_endpoint_post(
    query: SearchRequest,
    request: Request,
    stash_instance: Stash = Depends(get_stash),
):
    """POST: Hybrid semantic vector and keyword search across indexed documents with zerank-2 reranking."""
    if not query.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty.",
        )
    results = stash_instance.search_hybrid(
        query_text=query.query,
        top_k=query.limit or 10,
        hybrid_blend=query.hybrid_blend,
        as_search_results=True,
    )

    accept_header = request.headers.get("accept", "")
    wants_text = (query.format == "text") or (query.format is None and any(h in accept_header for h in ("text/plain", "text/markdown")))

    if wants_text:
        body = format_search_results_as_text(results)
        return PlainTextResponse(content=body, media_type="text/plain; charset=utf-8")

    return results
