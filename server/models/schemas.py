"""Pydantic schemas for Babel API requests and responses."""

from __future__ import annotations

from datetime import date as dt_date, datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, field_validator

from server.models.text import clean_normal_text


class SearchRequest(BaseModel):
    """Search query request payload."""
    query: str
    limit: int = 10
    """The percentage of blending keyword and vector search (0.0 = vector only, 1.0 = keyword only)."""
    hybrid_blend: float = 0.5
    """Response format: 'json' (default) or 'text' for RAG LLM context."""
    format: Optional[Literal["json", "text"]] = None


class WebSearchRequest(BaseModel):
    """Web search query request payload."""
    query: str
    limit: int = 5
    """Response format: 'json' (default) or 'text' for RAG LLM context."""
    format: Optional[Literal["json", "text"]] = None


class WebSearchResultItem(BaseModel):
    """Single web search result item from SearXNG."""
    title: str
    url: str
    content: str
    engine: Optional[str] = None
    score: Optional[float] = None


class SingleSearchResult(BaseModel):
    """Single search result item."""
    doc_id: int
    url: str
    content: str
    summary: str | None = None
    score: float
    date: Optional[dt_date] = None

    @field_validator("content", mode="before")
    @classmethod
    def clean_content_field(cls, v: Any) -> str:
        return clean_normal_text(str(v) if v is not None else "")


class DocumentDetailResponse(BaseModel):
    """Full content and metadata for a specific document."""
    doc_id: int
    url: str
    title: str = ""
    date: str | None = None
    summary: str | None = None
    content: str


class DocumentBatchRequest(BaseModel):
    """Batch document fetch request payload."""
    doc_ids: list[int] = []
    ids: Optional[list[int]] = None
    clean: bool = True
    format: Optional[Literal["json", "text"]] = None

    def resolved_ids(self) -> list[int]:
        """Resolve doc_ids from either 'doc_ids' or 'ids' field."""
        raw_list = self.doc_ids or self.ids or []
        clean: list[int] = []
        for x in raw_list:
            if isinstance(x, int):
                clean.append(x)
            elif str(x).isdigit():
                clean.append(int(str(x)))
        return list(dict.fromkeys(clean))


class DocumentSummary(BaseModel):
    """Lightweight Pydantic summary of a crawled document."""
    id: int
    url: str
    markdown: str | None = None
    status_code: int | None = None
    saved_at: datetime
