"""SQLAlchemy ORM models for Babel database tables."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Sequence

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from server.models.schemas import SingleSearchResult


class Base(DeclarativeBase):
    pass


class Document(Base):
    """
    SQLAlchemy ORM model representing a crawled document stored in the database.
    Corresponds to the 'storedcrawlresult' table.
    """
    __tablename__ = "storedcrawlresult"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    html: Mapped[str] = mapped_column(Text, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    media: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    links: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    downloaded_files: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    js_execution_result: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    screenshot: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    pdf: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_content: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    # Database column is "metadata". In DeclarativeBase, "metadata" is reserved for
    # MetaData, so we map column attribute as metadata_.
    metadata_: Mapped[str | None] = mapped_column("metadata", Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    response_headers: Mapped[str | None] = mapped_column(Text, nullable=True)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ssl_certificate: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    dispatch_result: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    redirected_url: Mapped[str | None] = mapped_column(String, nullable=True)
    redirected_status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    network_requests: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    console_messages: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    tables: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    head_fingerprint: Mapped[str | None] = mapped_column(String, nullable=True)
    cached_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    cache_status: Mapped[str | None] = mapped_column(String, nullable=True)
    crawl_stats: Mapped[str | None] = mapped_column(Text, nullable=True)
    saved_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    def __repr__(self) -> str:
        return f"<Document id={self.id} url={self.url!r} status={self.status_code}>"

    @property
    def meta(self) -> dict[str, Any]:
        """Parsed JSON metadata as a dictionary."""
        return self.parsed_metadata

    @property
    def parsed_metadata(self) -> dict[str, Any]:
        """Parsed dictionary from JSON metadata string."""
        if not self.metadata_:
            return {}
        try:
            return json.loads(self.metadata_)
        except (json.JSONDecodeError, TypeError):
            return {}

    @property
    def parsed_links(self) -> dict[str, Any] | list[Any]:
        """Parsed object from JSON links string."""
        if not self.links:
            return {}
        try:
            return json.loads(self.links)
        except (json.JSONDecodeError, TypeError):
            return {}

    @property
    def parsed_media(self) -> dict[str, Any] | list[Any]:
        """Parsed object from JSON media string."""
        if not self.media:
            return {}
        try:
            return json.loads(self.media)
        except (json.JSONDecodeError, TypeError):
            return {}

    @property
    def parsed_tables(self) -> list[Any]:
        """Parsed list from JSON tables string."""
        if not self.tables:
            return []
        try:
            return json.loads(self.tables)
        except (json.JSONDecodeError, TypeError):
            return []

    def to_dict(self, columns: Sequence[str] | None = None) -> dict[str, Any]:
        """Convert loaded document columns to a dictionary."""
        if columns is None:
            col_names = [c.name for c in self.__table__.columns]
        else:
            col_names = list(columns)

        res: dict[str, Any] = {}
        for col in col_names:
            attr = "metadata_" if col in ("metadata", "meta") else col
            dict_key = "metadata" if col in ("metadata", "metadata_") else col
            if hasattr(self, attr):
                try:
                    res[dict_key] = getattr(self, attr)
                except Exception:
                    pass
        return res

    def to_search_result(
        self,
        score: float = 0.0,
        summary: str | None = None,
    ) -> SingleSearchResult:
        """Helper to convert to SingleSearchResult Pydantic model."""
        content = self.markdown or self.html or ""
        doc_date = self.saved_at.date() if isinstance(self.saved_at, datetime) else date.today()
        return SingleSearchResult(
            doc_id=self.id,
            url=self.url,
            content=content,
            summary=summary,
            score=score,
            date=doc_date,
        )

    @classmethod
    def resolve_column(cls, name_or_col: Any):
        """Resolve a string name or Column attribute to the corresponding SQLAlchemy Column attribute."""
        if hasattr(name_or_col, "key") or hasattr(name_or_col, "name"):
            return name_or_col
        name = str(name_or_col).strip()
        if name in ("metadata", "meta"):
            return cls.metadata_
        if hasattr(cls, name):
            return getattr(cls, name)
        raise AttributeError(f"Document model has no column '{name}'. Available: {cls.column_names()}")

    @classmethod
    def column_names(cls) -> list[str]:
        """Return all column names of the underlying table."""
        return [c.name for c in cls.__table__.columns]


# Alias for crawl4ai compatibility
StoredCrawlResult = Document
