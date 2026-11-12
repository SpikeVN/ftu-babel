"""Search and Web Search routes for Babel."""

from __future__ import annotations

import logging
from typing import Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse
import httpx

from server.config import SEARXNG_URL
from server.models.schemas import (
    SearchRequest,
    SingleSearchResult,
    WebSearchRequest,
    WebSearchResultItem,
)
from server.models.text import (
    clean_normal_text,
    format_search_results_as_text,
    format_web_search_results_as_text,
)
from server.services.stash import Stash, get_stash

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Search"])


# -----------------------------------------------------------------------------
# Knowledge Base Search (/query & /search)
# -----------------------------------------------------------------------------

@router.get("/query", response_model=list[SingleSearchResult])
@router.get("/search", response_model=list[SingleSearchResult])
@router.get("/api/query", response_model=list[SingleSearchResult])
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


@router.post("/query", response_model=list[SingleSearchResult])
@router.post("/search", response_model=list[SingleSearchResult])
@router.post("/api/query", response_model=list[SingleSearchResult])
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


# -----------------------------------------------------------------------------
# Web Search (/web_search)
# -----------------------------------------------------------------------------

async def _fetch_searxng(query: str, limit: int = 5, timeout: float = 15.0) -> list[WebSearchResultItem]:
    """Execute search query against SearXNG instance."""
    if not query.strip():
        return []

    url = f"{SEARXNG_URL}/search"
    params = {"q": query.strip(), "format": "json"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, params=params)
            if r.status_code != 200:
                logger.warning("SearXNG returned HTTP %s: %s", r.status_code, r.text[:200])
                return []
            data = r.json()
            raw_results = data.get("results", [])
            items: list[WebSearchResultItem] = []
            for item in raw_results[:limit]:
                raw_content = item.get("content") or item.get("snippet") or ""
                items.append(
                    WebSearchResultItem(
                        title=clean_normal_text(item.get("title", "")),
                        url=item.get("url", ""),
                        content=clean_normal_text(raw_content),
                        engine=item.get("engine"),
                        score=item.get("score"),
                    )
                )
            return items
    except Exception as exc:
        logger.error("Failed to query SearXNG: %s", exc)
        return []


@router.get("/web_search", response_model=list[WebSearchResultItem])
@router.get("/api/web_search", response_model=list[WebSearchResultItem])
async def web_search_endpoint_get(
    request: Request,
    q: str = Query(..., description="Web search query string"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of results to return"),
    format: Optional[Literal["json", "text"]] = Query(None, description="Response format: 'json' (default) or 'text' for RAG LLM context"),
):
    """GET: Web search across search engines via SearXNG."""
    if not q.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Web search query cannot be empty.",
        )
    results = await _fetch_searxng(query=q, limit=limit)

    accept_header = request.headers.get("accept", "")
    wants_text = (format == "text") or (format is None and any(h in accept_header for h in ("text/plain", "text/markdown")))

    if wants_text:
        body = format_web_search_results_as_text(results)
        return PlainTextResponse(content=body, media_type="text/plain; charset=utf-8")

    return results


@router.post("/web_search", response_model=list[WebSearchResultItem])
@router.post("/api/web_search", response_model=list[WebSearchResultItem])
async def web_search_endpoint_post(
    payload: WebSearchRequest,
    request: Request,
):
    """POST: Web search across search engines via SearXNG."""
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Web search query cannot be empty.",
        )
    results = await _fetch_searxng(query=payload.query, limit=payload.limit or 5)

    accept_header = request.headers.get("accept", "")
    wants_text = (payload.format == "text") or (payload.format is None and any(h in accept_header for h in ("text/plain", "text/markdown")))

    if wants_text:
        body = format_web_search_results_as_text(results)
        return PlainTextResponse(content=body, media_type="text/plain; charset=utf-8")

    return results
