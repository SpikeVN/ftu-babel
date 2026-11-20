"""
stash.py - High-Performance SQLAlchemy API for Crawled Documents.

Provides an easy-to-use, column-selective API for querying documents from stash.db.
By requesting only specific columns (e.g., id, url, markdown), it avoids loading
gigabytes of raw HTML, binary screenshots, and blobs into memory.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import sqlite3
import threading
import urllib.parse
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import (
    Any,
    Generator,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    Union,
    overload,
)

from sqlalchemy import (
    Engine,
    Select,
    and_,
    asc,
    create_engine,
    desc,
    func,
    or_,
    select,
)
from sqlalchemy.orm import (
    Session,
    load_only,
    sessionmaker,
)
from sqlalchemy.sql.elements import ColumnElement

from server.config import (
    DEFAULT_DB_PATH,
    DEFAULT_EMBED_URL,
    FETCH_BATCH_MAX_CHARS,
    FETCH_DEFAULT_MAX_CHARS,
    FETCH_SHORT_DOC_CHARS,
    FTULABS_API_KEY,
    MOSS_API_KEY,
    MOSS_EMBEDDING_ENDPOINT,
    MOSS_RERANKING_ENDPOINT,
    RERANK_ENABLED,
    ZERANK_CANDIDATE_K,
    ZEROENTROPY_API_KEY,
)
from server.db import SQLiteConnectionPool
from server.models import Document, SingleSearchResult, StoredCrawlResult, clean_normal_text
from server.models.text import (
    _truncate_cleanly,
    clean_passage_lines,
    extract_centered_passage,
    extract_title_from_text,
    resolve_document_title,
)
from server.services.embedding import score_vector as _score_vector, search_similar as _search_similar
from server.services.rerank import rerank_hits

logger = logging.getLogger(__name__)

_ARTICLE_ID_RE = re.compile(r"/(\d+)-[^/]+$")


def _canonical_doc_key(url: str) -> str:
    """Normalize URL (decoding percent-encodings, stripping userinfo, collapsing FTU article IDs) for deduplication."""
    if not url:
        return ""
    decoded = urllib.parse.unquote(url).strip()
    parsed = urllib.parse.urlsplit(decoded)
    host = (parsed.hostname or parsed.netloc).lower()
    if parsed.port:
        host = f"{host}:{parsed.port}"
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    m = _ARTICLE_ID_RE.search(path)
    if m:
        return f"{host}#article:{m.group(1)}"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{path}{query}"

ColumnSpec = Union[str, ColumnElement[Any]]


def _find_default_db_path() -> str:
    """Locate the default stash SQLite database path."""
    return DEFAULT_DB_PATH

    candidates = [
        Path("stash.db"),
        Path("crawl_results.db"),
        Path("../stash.db"),
        Path(__file__).parent / "stash.db",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    return "stash.db"


class Stash:
    """
    SQLAlchemy interface for stash.db crawled documents.

    Provides granular, column-selective fetching so you only retrieve the columns
    you need without bloating memory with large HTML or binary payload fields.

    Usage:
        stash = Stash()
        # or as context manager:
        with Stash() as stash:
            urls = stash.get_column("url", limit=10)
            doc = stash.get(1, columns=["id", "url", "markdown"])
            doc_dict = stash.get(1, columns=["id", "url"], as_dict=True)
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        engine: Engine | None = None,
        echo: bool = False,
    ):
        resolved_path = str(db_path) if db_path else _find_default_db_path()
        self.db_path = resolved_path
        self._pool = SQLiteConnectionPool(self.db_path)

        if engine is not None:
            self.engine = engine
            self._owns_engine = False
        else:
            self.engine = create_engine(
                f"sqlite:///{resolved_path}",
                connect_args={"check_same_thread": False},
                echo=echo,
            )
            self._owns_engine = True

        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
        self._active_session: Session | None = None

    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Borrow a pre-warmed connection from the connection pool."""
        return self._pool.get_connection()

    # -------------------------------------------------------------------------
    # Context Management & Session Control
    # -------------------------------------------------------------------------

    def __enter__(self) -> "Stash":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close active sessions and dispose of internal engine if owned."""
        if self._active_session is not None:
            try:
                self._active_session.close()
            except Exception:
                pass
            self._active_session = None

        if hasattr(self, "_pool"):
            try:
                self._pool.close()
            except Exception:
                pass

        if self._owns_engine:
            try:
                self.engine.dispose()
            except Exception:
                pass

    @property
    def session(self) -> Session:
        """Get or create the active session for this Stash instance."""
        if self._active_session is None or not self._active_session.is_active:
            self._active_session = self.session_factory()
        return self._active_session

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        sess = self.session_factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    # -------------------------------------------------------------------------
    # Column Resolution Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def resolve_column(col: ColumnSpec) -> ColumnElement[Any]:
        """Resolve column name string or Column attribute to SQLAlchemy column."""
        return Document.resolve_column(col)

    @staticmethod
    def column_name(col: ColumnSpec) -> str:
        """Return canonical name string for a column."""
        if isinstance(col, str):
            c = col.strip()
            if c in ("metadata", "meta", "metadata_"):
                return "metadata"
            return c
        # For SQLAlchemy ColumnElement
        key = getattr(col, "key", None) or getattr(col, "name", None) or str(col)
        if key in ("metadata_", "metadata"):
            return "metadata"
        return key

    def _get_selectable(self, col: ColumnSpec) -> ColumnElement[Any]:
        """Return a selectable column element with label for mappings."""
        resolved = self.resolve_column(col)
        canonical = self.column_name(col)
        # Give column alias so mapped dict keys match canonical names (e.g. 'metadata')
        return resolved.label(canonical)

    def list_columns(self) -> list[str]:
        """Return all available column names in the document table."""
        return Document.column_names()

    @staticmethod
    def clean_html(html: str, base_url: str = "") -> str:
        """Convert raw HTML to cleaned Markdown using processing.cleanup rules."""
        from processing.cleanup import clean_html as _clean_html
        return _clean_html(html, base_url=base_url)

    @staticmethod
    def _resolve_keyword_parent_chunk(
        cursor: sqlite3.Cursor,
        doc_id: int,
        clean_snip: str,
        tokens: list[str],
    ) -> tuple[int | None, str | None]:
        """Locate the best matching parent chunk (chunk_index, chunk_text) for a keyword hit."""
        anchors: list[str] = []
        if clean_snip:
            segments: list[str] = []
            for part in clean_snip.replace("\n", "...").split("..."):
                seg = " ".join(part.split()).strip()
                if len(seg) >= 16:
                    segments.append(seg)
            segments.sort(key=len, reverse=True)
            for seg in segments[:2]:
                phrase = seg[:45]
                if len(seg) > 45:
                    sp = phrase.rfind(" ")
                    if sp >= 18:
                        phrase = phrase[:sp]
                phrase = phrase.strip()
                if phrase:
                    anchors.append(phrase)

        for tok in sorted((t.strip() for t in tokens if len(t.strip()) >= 2), key=len, reverse=True)[:2]:
            if tok not in anchors:
                anchors.append(tok)

        for anchor in anchors[:3]:
            row = cursor.execute(
                """
                SELECT chunk_index, chunk_text
                FROM document_chunks
                WHERE doc_id = ? AND chunk_type = 'parent'
                  AND (instr(lower(chunk_text), ?) > 0 OR instr(chunk_text, ?) > 0)
                ORDER BY chunk_index ASC
                LIMIT 1
                """,
                (doc_id, anchor.lower(), anchor),
            ).fetchone()
            if row is not None:
                return int(row[0]), str(row[1])
        return None, None

    def search_keyword(
        self,
        query_text: str,
        top_k: int = 20,
        conn: sqlite3.Connection | None = None,
    ) -> list[dict[str, Any]]:
        """Perform keyword BM25 search over fts_documents in SQLite with matched-chunk resolution."""
        clean = "".join(c if c.isalnum() or c.isspace() else " " for c in query_text).strip()
        tokens = clean.split()
        if not tokens:
            return []

        q_and = " ".join(f'"{t}"' for t in tokens)
        sql = """
        SELECT
            f.doc_id,
            f.url,
            f.title,
            bm25(f.fts_documents, 5.0, 2.0, 1.0, 3.0) AS bm25_rank,
            snippet(f.fts_documents, 3, '', '', '...', 64) AS snippet,
            COALESCE(ds.summary, '') AS summary,
            COALESCE(ds.published_date, substr(s.saved_at, 1, 10)) AS doc_date,
            s.saved_at
        FROM fts_documents f
        LEFT JOIN documents s ON s.id = f.doc_id
        LEFT JOIN document_summaries ds ON ds.doc_id = f.doc_id
        WHERE f.fts_documents MATCH ?
        ORDER BY bm25_rank ASC
        LIMIT ?
        """

        def _exec_search(c: sqlite3.Connection) -> tuple[list[Any], dict[int, int], dict[int, tuple[int | None, str | None]]]:
            cursor = c.cursor()
            rows = cursor.execute(sql, (q_and, top_k)).fetchall()
            if len(rows) < top_k and len(tokens) > 1:
                existing_ids = {r[0] for r in rows}
                q_or = " OR ".join(f'"{t}"' for t in tokens)
                additional = cursor.execute(sql, (q_or, top_k * 2)).fetchall()
                for r in additional:
                    if r[0] not in existing_ids:
                        rows.append(r)
                        existing_ids.add(r[0])
                        if len(rows) >= top_k:
                            break

            doc_ids = [int(r[0]) for r in rows]
            parent_counts: dict[int, int] = {}
            resolved_chunks: dict[int, tuple[int | None, str | None]] = {}
            if doc_ids:
                placeholders = ",".join("?" for _ in doc_ids)
                for did, cnt in cursor.execute(
                    f"SELECT doc_id, COUNT(*) FROM document_chunks "
                    f"WHERE chunk_type = 'parent' AND doc_id IN ({placeholders}) GROUP BY doc_id",
                    doc_ids,
                ).fetchall():
                    parent_counts[int(did)] = int(cnt)

                for r in rows:
                    did = int(r[0])
                    if parent_counts.get(did, 0) > 0:
                        raw_snip = r[4] or ""
                        clean_snip = clean_normal_text(raw_snip) if raw_snip else ""
                        resolved_chunks[did] = self._resolve_keyword_parent_chunk(
                            cursor, did, clean_snip, tokens
                        )
            return rows, parent_counts, resolved_chunks

        if conn is not None:
            rows, parent_counts, resolved_chunks = _exec_search(conn)
        else:
            with self.get_connection() as c:
                rows, parent_counts, resolved_chunks = _exec_search(c)

        hits = []
        for r in rows:
            doc_id, url, title, _bm25_rank, snippet, summary, doc_date, saved_at = r
            doc_id_int = int(doc_id)
            total_chunks = parent_counts.get(doc_id_int, 0)
            clean_snip = clean_normal_text(snippet) if snippet else ""
            chunk_idx, parent_chunk_text = resolved_chunks.get(doc_id_int, (None, None))

            if chunk_idx is not None and parent_chunk_text:
                centered = extract_centered_passage(
                    parent_chunk_text,
                    clean_snip or query_text,
                    max_chars=1200,
                )
                effective_snippet = centered or clean_snip
                effective_rerank = centered or clean_snip or (summary.strip() if summary else "")
            else:
                effective_snippet = clean_snip
                effective_rerank = clean_snip or (summary.strip() if summary else "")

            resolved_title = resolve_document_title(
                title or "",
                effective_snippet or "",
                url or "",
            )
            hits.append({
                "doc_id": doc_id_int,
                "url": url,
                "title": resolved_title,
                "snippet": effective_snippet,
                "summary": summary.strip() if summary else "",
                "published_date": str(doc_date) if doc_date else "",
                "saved_at": str(saved_at) if saved_at else "",
                "rerank_text": effective_rerank,
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks if total_chunks > 0 else None,
            })
        return hits

    def search_hybrid(
        self,
        query_text: str,
        top_k: int = 10,
        hybrid_blend: float = 0.5,
        embed_url: str | None = None,
        as_search_results: bool = False,
    ) -> list[dict[str, Any]] | list[Any]:
        """
        Perform hybrid vector (sqlite-vec) and keyword (FTS5 BM25) search,
        combining candidate rankings with Reciprocal Rank Fusion (RRF),
        and reranking candidates using zerank-2.
        """
        has_rerank_key = RERANK_ENABLED and float(hybrid_blend) > 0.0
        candidate_k = max(top_k * 2, ZERANK_CANDIDATE_K) if has_rerank_key else max(top_k * 2, 20)
        raw_candidate_k = max(candidate_k * 2, 50)

        vec_hits: list[dict[str, Any]] = []
        kw_hits: list[dict[str, Any]] = []

        # Single pooled connection for all DB operations in this request
        with self.get_connection() as conn:
            # 1. Retrieve vector candidates (when hybrid_blend > 0.0)
            if hybrid_blend > 0.0:
                try:
                    vec_hits = _search_similar(
                        query_text,
                        top_k=raw_candidate_k,
                        db_path=self.db_path,
                        embed_url=embed_url or DEFAULT_EMBED_URL,
                        include_rerank_text=has_rerank_key,
                        conn=conn,
                    )
                except Exception as exc:
                    logger.warning("Vector search failed (%s); continuing with keyword results.", exc)

            # 2. Retrieve keyword candidates (when hybrid_blend < 1.0)
            if hybrid_blend < 1.0:
                try:
                    kw_hits = self.search_keyword(
                        query_text,
                        top_k=raw_candidate_k,
                        conn=conn,
                    )
                except Exception as exc:
                    logger.warning("Keyword FTS search failed (%s); continuing with vector results.", exc)

            # 3. Fuse rankings with Reciprocal Rank Fusion (RRF)
            w_vec = max(0.0, min(1.0, float(hybrid_blend)))
            w_kw = 1.0 - w_vec

            rrf_scores: dict[int, float] = {}
            candidate_docs: dict[int, dict[str, Any]] = {}
            RRF_K = 60.0

            for rank, hit in enumerate(vec_hits):
                doc_id = hit["doc_id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + w_vec * (1.0 / (RRF_K + rank + 1))
                if doc_id not in candidate_docs:
                    candidate_docs[doc_id] = dict(hit)

            for rank, hit in enumerate(kw_hits):
                doc_id = hit["doc_id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + w_kw * (1.0 / (RRF_K + rank + 1))
                if doc_id not in candidate_docs:
                    candidate_docs[doc_id] = dict(hit)
                else:
                    if not candidate_docs[doc_id].get("snippet") and hit.get("snippet"):
                        candidate_docs[doc_id]["snippet"] = hit["snippet"]
                    if not candidate_docs[doc_id].get("summary") and hit.get("summary"):
                        candidate_docs[doc_id]["summary"] = hit["summary"]
                    if not candidate_docs[doc_id].get("published_date") and hit.get("published_date"):
                        candidate_docs[doc_id]["published_date"] = hit["published_date"]
                    if not candidate_docs[doc_id].get("rerank_text") and hit.get("rerank_text"):
                        candidate_docs[doc_id]["rerank_text"] = hit["rerank_text"]
                    if candidate_docs[doc_id].get("chunk_index") is None and hit.get("chunk_index") is not None:
                        candidate_docs[doc_id]["chunk_index"] = hit["chunk_index"]
                    if not candidate_docs[doc_id].get("total_chunks") and hit.get("total_chunks"):
                        candidate_docs[doc_id]["total_chunks"] = hit["total_chunks"]

            fused = sorted(candidate_docs.keys(), key=lambda did: rrf_scores[did], reverse=True)
            max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
            hits: list[dict[str, Any]] = []
            seen_canon_keys: set[str] = set()
            seen_snippets: set[str] = set()
            for did in fused:
                doc_item = candidate_docs[did]
                canon_key = _canonical_doc_key(doc_item.get("url", ""))
                if canon_key and canon_key in seen_canon_keys:
                    continue
                snip_key = (doc_item.get("snippet") or "").strip()[:200].lower()
                if snip_key and snip_key in seen_snippets:
                    continue
                if canon_key:
                    seen_canon_keys.add(canon_key)
                if snip_key:
                    seen_snippets.add(snip_key)
                doc_item["score"] = round(rrf_scores[did] / max_rrf, 4) if max_rrf > 0 else 0.5
                hits.append(doc_item)
                if len(hits) >= candidate_k:
                    break

            # 4. Rerank candidates with zerank-2
            if has_rerank_key and hits:
                try:
                    hits = rerank_hits(query_text, hits, top_k=top_k)
                except Exception as exc:
                    logger.warning("Reranking failed (%s); falling back to hybrid RRF order.", exc)
                    hits = hits[:top_k]
            else:
                hits = hits[:top_k]

            # Ensure summary is present for all returned hits (reusing the open conn)
            missing_summary_ids = [h["doc_id"] for h in hits if not h.get("summary")]
            if missing_summary_ids:
                placeholders = ",".join("?" for _ in missing_summary_ids)
                summary_rows = conn.execute(
                    f"SELECT doc_id, summary, published_date FROM document_summaries WHERE doc_id IN ({placeholders})",
                    missing_summary_ids,
                ).fetchall()
                summary_data = {r[0]: (r[1], r[2]) for r in summary_rows}
                for h in hits:
                    did = h["doc_id"]
                    if did in summary_data:
                        sum_text, p_date = summary_data[did]
                        if sum_text and not h.get("summary"):
                            h["summary"] = sum_text
                        if p_date and not h.get("published_date"):
                            h["published_date"] = p_date

        if not as_search_results:
            return hits

        # 5. Convert to SingleSearchResult models
        out: list[SingleSearchResult] = []
        for h in hits:
            raw_date = h.get("published_date") or h.get("saved_at", "")
            try:
                dt = date.fromisoformat(str(raw_date)[:10]) if raw_date else date.today()
            except Exception:
                dt = date.today()
            resolved_title = resolve_document_title(
                h.get("title") or "",
                h.get("rerank_text") or h.get("snippet") or "",
                h.get("url", ""),
            )
            out.append(
                SingleSearchResult(
                    doc_id=h["doc_id"],
                    title=resolved_title,
                    url=h["url"],
                    content=h.get("snippet", ""),
                    summary=h.get("summary"),
                    score=float(h.get("score", 0.0)),
                    date=dt,
                    chunk_index=h.get("chunk_index"),
                    total_chunks=h.get("total_chunks"),
                )
            )
        return out

    def search_similar(
        self,
        query_text: str,
        top_k: int = 10,
        hybrid_blend: float = 0.5,
        embed_url: str | None = None,
        as_search_results: bool = False,
    ) -> list[dict[str, Any]] | list[Any]:
        """Perform hybrid search (vector + keyword + zerank-2 rerank)."""
        return self.search_hybrid(
            query_text=query_text,
            top_k=top_k,
            hybrid_blend=hybrid_blend,
            embed_url=embed_url,
            as_search_results=as_search_results,
        )

    def score_vector(
        self,
        vector: Sequence[float],
        top_k: int = 10,
        as_search_results: bool = False,
    ) -> list[dict[str, Any]] | list[Any]:
        """Score documents against an arbitrary 768-dim vector using sqlite-vec."""
        hits = _score_vector(vector, top_k=top_k, db_path=self.db_path)
        if not as_search_results:
            return hits

        out: list[SingleSearchResult] = []
        for h in hits:
            raw_date = h.get("saved_at", "")
            try:
                dt = date.fromisoformat(raw_date[:10]) if raw_date else date.today()
            except Exception:
                dt = date.today()
            out.append(
                SingleSearchResult(
                    doc_id=h["doc_id"],
                    url=h["url"],
                    content=h.get("snippet", ""),
                    score=h["score"],
                    date=dt,
                )
            )
        return out

    # -------------------------------------------------------------------------
    # Single Document Queries
    # -------------------------------------------------------------------------

    def get(
        self,
        id_or_url: int | str,
        columns: Sequence[ColumnSpec] | ColumnSpec | None = None,
        as_dict: bool = False,
    ) -> Document | dict[str, Any] | None:
        """
        Fetch a document by ID (integer or numeric string) or by URL.

        Args:
            id_or_url: Document ID (e.g. 1) or full URL (e.g. 'https://...').
            columns: Specific column(s) to fetch. If None, loads all columns.
            as_dict: If True, returns a plain dictionary instead of a Document object.

        Returns:
            Document object, dict, or None if not found.
        """
        # Determine filter
        if isinstance(id_or_url, int) or (isinstance(id_or_url, str) and id_or_url.isdigit()):
            condition = Document.id == int(id_or_url)
        else:
            condition = Document.url == str(id_or_url)

        # Normalize columns parameter
        col_list: list[ColumnSpec] | None = None
        if columns is not None:
            col_list = [columns] if isinstance(columns, (str, ColumnElement)) else list(columns)

        sess = self.session

        if as_dict:
            if col_list:
                selectables = [self._get_selectable(c) for c in col_list]
                stmt = select(*selectables).where(condition).limit(1)
                row = sess.execute(stmt).mappings().first()
                return dict(row) if row is not None else None
            else:
                doc = sess.scalars(select(Document).where(condition).limit(1)).first()
                return doc.to_dict() if doc is not None else None

        # Return ORM Document
        if col_list:
            resolved_cols = [self.resolve_column(c) for c in col_list]
            stmt = select(Document).options(load_only(*resolved_cols)).where(condition).limit(1)
        else:
            stmt = select(Document).where(condition).limit(1)

        return sess.scalars(stmt).first()

    def get_by_id(
        self,
        doc_id: int,
        columns: Sequence[ColumnSpec] | ColumnSpec | None = None,
        as_dict: bool = False,
    ) -> Document | dict[str, Any] | None:
        """Fetch document by primary key ID."""
        return self.get(doc_id, columns=columns, as_dict=as_dict)

    def get_document_detail(
        self,
        doc_id: int,
        clean: bool = True,
        chunk_index: int | None = None,
        window: int = 1,
        max_chars: int | None = None,
    ) -> dict[str, Any] | None:
        """Fetch document content (full for short docs, windowed chunks or budgeted head for long docs) and metadata."""
        docs = self.get_documents_detail(
            [doc_id],
            clean=clean,
            chunk_index=chunk_index,
            window=window,
            max_chars=max_chars,
        )
        return docs[0] if docs else None

    def get_documents_detail(
        self,
        doc_ids: Sequence[int],
        clean: bool = True,
        chunk_index: int | None = None,
        window: int = 1,
        max_chars: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch document content and metadata for multiple doc_ids with chunk-windowing and batch context protection.
        Short documents return in full; long documents with `chunk_index` return surrounding neighbor chunks plus
        a Chunk 0 header preview; long un-indexed documents return a budgeted head with chunk-navigation guidance.
        """
        if not doc_ids:
            return []

        clean_ids = list(
            dict.fromkeys(
                int(i) for i in doc_ids if isinstance(i, int) or str(i).strip().isdigit()
            )
        )
        if not clean_ids:
            return []

        win = max(0, min(int(1 if window is None else window), 5))
        batch_cap = FETCH_BATCH_MAX_CHARS
        requested_cap = int(max_chars) if (max_chars is not None and int(max_chars) > 0) else FETCH_DEFAULT_MAX_CHARS
        per_doc_base = min(requested_cap, batch_cap, max(6000, batch_cap // max(1, len(clean_ids))))
        remaining_budget = batch_cap

        placeholders = ",".join("?" for _ in clean_ids)
        sql = f"""
        SELECT
            s.id,
            s.url,
            COALESCE(json_extract(s.metadata, '$.title'), '') AS title,
            COALESCE(ds.summary, '') AS summary,
            COALESCE(ds.published_date, substr(s.saved_at, 1, 10)) AS published_date,
            COALESCE(s.markdown, s.html, '') AS content
        FROM documents s
        LEFT JOIN document_summaries ds ON ds.doc_id = s.id
        WHERE s.id IN ({placeholders})
        """
        count_sql = f"""
        SELECT doc_id, COUNT(*)
        FROM document_chunks
        WHERE chunk_type = 'parent' AND doc_id IN ({placeholders})
        GROUP BY doc_id
        """

        results: list[dict[str, Any]] = []
        with self.get_connection() as conn:
            rows = conn.execute(sql, clean_ids).fetchall()
            raw_by_id = {int(r[0]): r for r in rows}
            count_rows = conn.execute(count_sql, clean_ids).fetchall()
            parent_counts = {int(did): int(cnt) for did, cnt in count_rows}

            for did in clean_ids:
                row = raw_by_id.get(did)
                if row is None:
                    continue
                id_, url, title, summary, pub_date, raw_content = row
                raw_text = raw_content or ""
                resolved_title = resolve_document_title(title or "", raw_text[:2000], url or "")
                full_text = clean_normal_text(raw_text) if clean else raw_text
                total_chars = len(full_text)
                total_chunks = parent_counts.get(did, 0)
                doc_cap = max(4000, min(per_doc_base, remaining_budget))

                # Case 1: Short document fits in full
                if total_chars <= min(FETCH_SHORT_DOC_CHARS, doc_cap) or (
                    chunk_index is None and total_chars <= doc_cap
                ):
                    final_content = full_text
                    resp_chunk_index = chunk_index
                    chunk_start = 0 if total_chunks > 0 else None
                    chunk_end = (total_chunks - 1) if total_chunks > 0 else None
                    truncated = False

                # Case 2: Windowed chunk fetch around target chunk_index
                elif chunk_index is not None and total_chunks > 0:
                    ci = max(0, min(int(chunk_index), total_chunks - 1))
                    c_start = max(0, ci - win)
                    c_end = min(total_chunks - 1, ci + win)

                    chunk_rows = conn.execute(
                        """
                        SELECT chunk_index, chunk_text
                        FROM document_chunks
                        WHERE doc_id = ? AND chunk_type = 'parent' AND chunk_index BETWEEN ? AND ?
                        ORDER BY chunk_index ASC
                        """,
                        (did, c_start, c_end),
                    ).fetchall()

                    parts: list[str] = []
                    if c_start > 0:
                        head_row = conn.execute(
                            """
                            SELECT chunk_text
                            FROM document_chunks
                            WHERE doc_id = ? AND chunk_type = 'parent' AND chunk_index = 0
                            LIMIT 1
                            """,
                            (did,),
                        ).fetchone()
                        if head_row and head_row[0]:
                            head_text = clean_passage_lines(head_row[0]) if clean else str(head_row[0])
                            head_preview = _truncate_cleanly(head_text.strip(), 600)
                            if head_preview:
                                parts.append(f"[Document Header / Chunk 0 preview]\n{head_preview}")

                    for c_idx, c_text in chunk_rows:
                        body = (clean_normal_text(c_text) if clean else str(c_text)).strip()
                        marker = " (matched)" if int(c_idx) == ci else ""
                        parts.append(f"[Chunk {int(c_idx)} of 0..{total_chunks - 1}]{marker}\n{body}")

                    assembled = "\n\n---\n\n".join(parts) if parts else full_text
                    is_partial = c_start > 0 or c_end < (total_chunks - 1)

                    footer = ""
                    if is_partial:
                        nav_hints: list[str] = []
                        if c_start > 0:
                            prev_idx = max(0, c_start - win - 1)
                            nav_hints.append(f"earlier (0..{c_start - 1}): chunk_index={prev_idx}")
                        if c_end < total_chunks - 1:
                            next_idx = min(total_chunks - 1, c_end + win + 1)
                            nav_hints.append(f"later ({c_end + 1}..{total_chunks - 1}): chunk_index={next_idx}")
                        footer = (
                            f"\n\n[Showing chunks {c_start}..{c_end} of 0..{total_chunks - 1}. "
                            f"Call babel_fetch(doc_ids={did}, chunk_index=..., window={win}) for "
                            + "; ".join(nav_hints)
                            + ".]"
                        )

                    cap_trimmed = False
                    if len(assembled) + len(footer) > doc_cap:
                        allow = max(1000, doc_cap - len(footer) - 4)
                        assembled = _truncate_cleanly(assembled, allow)
                        cap_trimmed = True

                    final_content = f"{assembled}{footer}" if footer else assembled
                    resp_chunk_index = ci
                    chunk_start = c_start
                    chunk_end = c_end
                    truncated = is_partial or cap_trimmed

                # Case 3: Long document without chunk_index (or unchunked doc) -> budgeted head + hint
                else:
                    footer_reserve = 220 if total_chunks > 0 else 80
                    text_cap = max(1000, doc_cap - footer_reserve)
                    cut = full_text[:text_cap]
                    if len(full_text) > text_cap:
                        tail_floor = max(len(cut) // 2, len(cut) - 200)
                        last_ws = max(cut.rfind("\n", tail_floor), cut.rfind(" ", tail_floor))
                        if last_ws > len(cut) // 2:
                            cut = cut[:last_ws]
                    cut = cut.rstrip()

                    if total_chunks > 0:
                        footer = (
                            f"\n\n[Truncated at {len(cut):,}/{total_chars:,} chars "
                            f"({total_chunks} chunks: 0..{total_chunks - 1}). "
                            f"Call babel_fetch(doc_ids={did}, chunk_index=<N>, window=1) "
                            f"using the chunk_index from babel_query to read a specific section without truncation.]"
                        )
                    else:
                        footer = f"\n\n[Truncated at {len(cut):,}/{total_chars:,} chars to fit context window.]"

                    final_content = f"{cut}{footer}"
                    resp_chunk_index = None
                    chunk_start = 0 if total_chunks > 0 else None
                    chunk_end = None
                    truncated = True

                remaining_budget = max(0, remaining_budget - len(final_content))
                results.append({
                    "doc_id": int(id_),
                    "url": url,
                    "title": resolved_title,
                    "summary": summary if summary else None,
                    "date": str(pub_date) if pub_date else None,
                    "content": final_content,
                    "chunk_index": resp_chunk_index,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "total_chunks": total_chunks if total_chunks > 0 else None,
                    "truncated": truncated,
                    "total_chars": total_chars,
                })

        return results

    def get_by_url(
        self,
        url: str,
        columns: Sequence[ColumnSpec] | ColumnSpec | None = None,
        as_dict: bool = False,
    ) -> Document | dict[str, Any] | None:
        """Fetch document by unique URL."""
        return self.get(url, columns=columns, as_dict=as_dict)

    # -------------------------------------------------------------------------
    # Single Column Extraction (Pluck across multiple rows)
    # -------------------------------------------------------------------------

    def get_column(
        self,
        column: ColumnSpec,
        *,
        limit: int | None = None,
        offset: int = 0,
        success_only: bool = False,
        distinct: bool = False,
        order_by: ColumnSpec | None = None,
        descending: bool = False,
        where: Any = None,
    ) -> list[Any]:
        """
        Pluck and return a single column's values as a flat list across documents.

        Example:
            urls = stash.get_column("url", limit=100)
            ids = stash.get_column("id")
            markdowns = stash.get_column("markdown", limit=10)
        """
        col = self.resolve_column(column)
        target = col.distinct() if distinct else col
        stmt = select(target)

        if success_only:
            stmt = stmt.where(Document.success == True)  # noqa: E712
        if where is not None:
            stmt = stmt.where(where)

        if order_by is not None:
            order_col = self.resolve_column(order_by)
            stmt = stmt.order_by(desc(order_col) if descending else asc(order_col))

        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        return list(self.session.scalars(stmt).all())

    # -------------------------------------------------------------------------
    # Multiple Columns Extraction
    # -------------------------------------------------------------------------

    def get_columns(
        self,
        *columns: ColumnSpec,
        limit: int | None = None,
        offset: int = 0,
        success_only: bool = False,
        as_dict: bool = True,
        order_by: ColumnSpec | None = None,
        descending: bool = False,
        where: Any = None,
    ) -> list[dict[str, Any]] | list[tuple[Any, ...]]:
        """
        Fetch specific columns for multiple documents efficiently.

        Example:
            records = stash.get_columns("id", "url", "status_code", limit=10)
            # returns: [{"id": 1, "url": "...", "status_code": 200}, ...]
        """
        if not columns:
            columns = ("id", "url")

        selectables = [self._get_selectable(c) for c in columns]
        stmt = select(*selectables)

        if success_only:
            stmt = stmt.where(Document.success == True)  # noqa: E712
        if where is not None:
            stmt = stmt.where(where)

        if order_by is not None:
            order_col = self.resolve_column(order_by)
            stmt = stmt.order_by(desc(order_col) if descending else asc(order_col))

        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        if as_dict:
            rows = self.session.execute(stmt).mappings().all()
            return [dict(r) for r in rows]
        else:
            return list(self.session.execute(stmt).all())

    # -------------------------------------------------------------------------
    # Key-to-Value Mapping Extraction
    # -------------------------------------------------------------------------

    def get_map(
        self,
        key_column: ColumnSpec,
        value_column: ColumnSpec,
        *,
        limit: int | None = None,
        offset: int = 0,
        success_only: bool = False,
        where: Any = None,
    ) -> dict[Any, Any]:
        """
        Extract two columns directly into a Python dictionary mapping {key: value}.

        Example:
            id_to_url = stash.get_map("id", "url", limit=1000)
            url_to_id = stash.get_map("url", "id")
        """
        k_col = self.resolve_column(key_column)
        v_col = self.resolve_column(value_column)
        stmt = select(k_col, v_col)

        if success_only:
            stmt = stmt.where(Document.success == True)  # noqa: E712
        if where is not None:
            stmt = stmt.where(where)
        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        rows = self.session.execute(stmt).all()
        return {r[0]: r[1] for r in rows}

    # -------------------------------------------------------------------------
    # Document List Queries (with Selective Column Loading)
    # -------------------------------------------------------------------------

    def get_documents(
        self,
        columns: Sequence[ColumnSpec] | ColumnSpec | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
        success_only: bool = False,
        status_code: int | None = None,
        url_contains: str | None = None,
        order_by: ColumnSpec = "id",
        descending: bool = False,
        as_dict: bool = False,
    ) -> list[Document] | list[dict[str, Any]]:
        """
        Query multiple documents with selective column loading and flexible filters.

        Args:
            columns: Columns to load (e.g. ['id', 'url', 'markdown']).
            limit: Maximum rows to return.
            offset: Rows to skip.
            success_only: Only return documents with success=True.
            status_code: Filter by HTTP status code (e.g. 200).
            url_contains: Substring match on URL.
            order_by: Column to sort by (default 'id').
            descending: Sort order.
            as_dict: If True, return list of dicts instead of Document objects.
        """
        col_list: list[ColumnSpec] | None = None
        if columns is not None:
            col_list = [columns] if isinstance(columns, (str, ColumnElement)) else list(columns)

        filters = []
        if success_only:
            filters.append(Document.success == True)  # noqa: E712
        if status_code is not None:
            filters.append(Document.status_code == status_code)
        if url_contains:
            filters.append(Document.url.contains(url_contains))

        order_col = self.resolve_column(order_by)
        ordering = desc(order_col) if descending else asc(order_col)

        sess = self.session

        if as_dict:
            if col_list:
                selectables = [self._get_selectable(c) for c in col_list]
            else:
                selectables = [self._get_selectable(c) for c in Document.column_names()]
            stmt = select(*selectables)
            if filters:
                stmt = stmt.where(and_(*filters))
            stmt = stmt.order_by(ordering).offset(offset).limit(limit)
            rows = sess.execute(stmt).mappings().all()
            return [dict(r) for r in rows]

        # Return ORM Document instances
        stmt = select(Document)
        if col_list:
            resolved_cols = [self.resolve_column(c) for c in col_list]
            stmt = stmt.options(load_only(*resolved_cols))
        if filters:
            stmt = stmt.where(and_(*filters))
        stmt = stmt.order_by(ordering).offset(offset).limit(limit)

        return list(sess.scalars(stmt).all())

    # -------------------------------------------------------------------------
    # Streaming / Chunked Iterators (Memory-Safe for 4GB+ Databases)
    # -------------------------------------------------------------------------

    def iter_documents(
        self,
        columns: Sequence[ColumnSpec] | ColumnSpec | None = None,
        *,
        batch_size: int = 500,
        success_only: bool = False,
        status_code: int | None = None,
        order_by: ColumnSpec = "id",
        descending: bool = False,
    ) -> Iterator[Document]:
        """
        Stream documents in batches using load_only to keep memory usage low.

        Example:
            for doc in stash.iter_documents(columns=["id", "markdown"], batch_size=500):
                process(doc.id, doc.markdown)
        """
        col_list: list[ColumnSpec] | None = None
        if columns is not None:
            col_list = [columns] if isinstance(columns, (str, ColumnElement)) else list(columns)

        filters = []
        if success_only:
            filters.append(Document.success == True)  # noqa: E712
        if status_code is not None:
            filters.append(Document.status_code == status_code)

        order_col = self.resolve_column(order_by)
        ordering = desc(order_col) if descending else asc(order_col)

        stmt = select(Document)
        if col_list:
            resolved_cols = [self.resolve_column(c) for c in col_list]
            stmt = stmt.options(load_only(*resolved_cols))
        if filters:
            stmt = stmt.where(and_(*filters))
        stmt = stmt.order_by(ordering).execution_options(yield_per=batch_size)

        sess = self.session
        for doc in sess.scalars(stmt):
            yield doc

    def iter_records(
        self,
        columns: Sequence[ColumnSpec] | ColumnSpec,
        *,
        batch_size: int = 500,
        success_only: bool = False,
        status_code: int | None = None,
        order_by: ColumnSpec = "id",
        descending: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """
        Stream lightweight row dictionaries in batches.

        Example:
            for row in stash.iter_records(columns=["id", "url", "saved_at"], batch_size=1000):
                print(row["url"])
        """
        col_list = [columns] if isinstance(columns, (str, ColumnElement)) else list(columns)
        selectables = [self._get_selectable(c) for c in col_list]

        filters = []
        if success_only:
            filters.append(Document.success == True)  # noqa: E712
        if status_code is not None:
            filters.append(Document.status_code == status_code)

        order_col = self.resolve_column(order_by)
        ordering = desc(order_col) if descending else asc(order_col)

        stmt = select(*selectables)
        if filters:
            stmt = stmt.where(and_(*filters))
        stmt = stmt.order_by(ordering).execution_options(yield_per=batch_size)

        sess = self.session
        for mapping in sess.execute(stmt).mappings():
            yield dict(mapping)

    # -------------------------------------------------------------------------
    # Dedicated Column Value Shortcuts
    # -------------------------------------------------------------------------

    def get_field(self, id_or_url: int | str, column: ColumnSpec) -> Any:
        """Fetch a single field value for a specific document without loading anything else."""
        res = self.get(id_or_url, columns=[column], as_dict=True)
        if res is None:
            return None
        canon = self.column_name(column)
        return res.get(canon)

    def get_markdown(self, id_or_url: int | str) -> str | None:
        """Fetch only the markdown text of a document."""
        return self.get_field(id_or_url, "markdown")

    def get_html(self, id_or_url: int | str) -> str | None:
        """Fetch only the raw HTML text of a document."""
        return self.get_field(id_or_url, "html")

    def get_metadata(self, id_or_url: int | str) -> dict[str, Any] | None:
        """Fetch and parse JSON metadata for a document."""
        raw = self.get_field(id_or_url, "metadata")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def get_links(self, id_or_url: int | str) -> dict[str, Any] | list[Any] | None:
        """Fetch and parse JSON links for a document."""
        raw = self.get_field(id_or_url, "links")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def get_media(self, id_or_url: int | str) -> dict[str, Any] | list[Any] | None:
        """Fetch and parse JSON media for a document."""
        raw = self.get_field(id_or_url, "media")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def get_tables(self, id_or_url: int | str) -> list[Any] | None:
        """Fetch and parse JSON tables for a document."""
        raw = self.get_field(id_or_url, "tables")
        if not raw:
            return []
        try:
            return json.loads(raw)
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # Convenience & Inspection Helpers
    # -------------------------------------------------------------------------

    def count(
        self,
        *,
        success_only: bool = False,
        url_contains: str | None = None,
        status_code: int | None = None,
    ) -> int:
        """Count total matching documents in the stash."""
        stmt = select(func.count(Document.id))
        filters = []
        if success_only:
            filters.append(Document.success == True)  # noqa: E712
        if status_code is not None:
            filters.append(Document.status_code == status_code)
        if url_contains:
            filters.append(Document.url.contains(url_contains))
        if filters:
            stmt = stmt.where(and_(*filters))
        return int(self.session.scalar(stmt) or 0)

    def exists(self, id_or_url: int | str) -> bool:
        """Check whether a document exists by ID or URL."""
        if isinstance(id_or_url, int) or (isinstance(id_or_url, str) and id_or_url.isdigit()):
            condition = Document.id == int(id_or_url)
        else:
            condition = Document.url == str(id_or_url)
        stmt = select(Document.id).where(condition).limit(1)
        return self.session.scalar(stmt) is not None

    def search_url(
        self,
        query: str,
        *,
        columns: Sequence[ColumnSpec] | None = None,
        limit: int = 50,
        as_dict: bool = True,
    ) -> list[Any]:
        """Search documents by URL substring."""
        cols = columns or ["id", "url"]
        return self.get_documents(
            columns=cols,
            url_contains=query,
            limit=limit,
            as_dict=as_dict,
        )

    # -------------------------------------------------------------------------
    # Mutation Helpers (Add, Update, Delete)
    # -------------------------------------------------------------------------

    def add(self, document_or_dict: Document | dict[str, Any]) -> Document:
        """Add and commit a new document record."""
        with self.session_scope() as sess:
            if isinstance(document_or_dict, dict):
                data = dict(document_or_dict)
                if "metadata" in data and "metadata_" not in data:
                    data["metadata_"] = data.pop("metadata")
                doc = Document(**data)
            else:
                doc = document_or_dict
            sess.add(doc)
            sess.flush()
            sess.refresh(doc)
            return doc

    def update(self, id_or_url: int | str, **fields: Any) -> bool:
        """Update fields on an existing document."""
        with self.session_scope() as sess:
            if isinstance(id_or_url, int) or (isinstance(id_or_url, str) and id_or_url.isdigit()):
                condition = Document.id == int(id_or_url)
            else:
                condition = Document.url == str(id_or_url)

            doc = sess.scalars(select(Document).where(condition).limit(1)).first()
            if not doc:
                return False

            for k, v in fields.items():
                attr = "metadata_" if k in ("metadata", "meta") else k
                if hasattr(doc, attr):
                    setattr(doc, attr, v)
            return True

    def delete(self, id_or_url: int | str) -> bool:
        """Delete a document by ID or URL."""
        with self.session_scope() as sess:
            if isinstance(id_or_url, int) or (isinstance(id_or_url, str) and id_or_url.isdigit()):
                condition = Document.id == int(id_or_url)
            else:
                condition = Document.url == str(id_or_url)

            doc = sess.scalars(select(Document).where(condition).limit(1)).first()
            if not doc:
                return False
            sess.delete(doc)
            return True


# -----------------------------------------------------------------------------
# FastAPI Dependency Provider
# -----------------------------------------------------------------------------

def get_stash() -> Stash:
    """FastAPI dependency for injecting the shared Stash instance into route handlers."""
    return get_default_stash()


# -----------------------------------------------------------------------------
# Module-Level Convenience Functions (using a lazily initialized default stash)
# -----------------------------------------------------------------------------

_default_stash: Stash | None = None


def get_default_stash() -> Stash:
    """Return or initialize the module-level default Stash instance."""
    global _default_stash
    if _default_stash is None:
        _default_stash = Stash()
    return _default_stash


def get(
    id_or_url: int | str,
    columns: Sequence[ColumnSpec] | ColumnSpec | None = None,
    as_dict: bool = False,
) -> Document | dict[str, Any] | None:
    """Fetch a document by ID or URL using the default Stash instance."""
    return get_default_stash().get(id_or_url, columns=columns, as_dict=as_dict)


def get_by_id(
    doc_id: int,
    columns: Sequence[ColumnSpec] | ColumnSpec | None = None,
    as_dict: bool = False,
) -> Document | dict[str, Any] | None:
    """Fetch document by primary key ID using default Stash instance."""
    return get_default_stash().get_by_id(doc_id, columns=columns, as_dict=as_dict)


def get_by_url(
    url: str,
    columns: Sequence[ColumnSpec] | ColumnSpec | None = None,
    as_dict: bool = False,
) -> Document | dict[str, Any] | None:
    """Fetch document by unique URL using default Stash instance."""
    return get_default_stash().get_by_url(url, columns=columns, as_dict=as_dict)


def get_column(
    column: ColumnSpec,
    *,
    limit: int | None = None,
    offset: int = 0,
    success_only: bool = False,
    distinct: bool = False,
    order_by: ColumnSpec | None = None,
    descending: bool = False,
    where: Any = None,
) -> list[Any]:
    """Pluck a single column as a flat list using default Stash instance."""
    return get_default_stash().get_column(
        column,
        limit=limit,
        offset=offset,
        success_only=success_only,
        distinct=distinct,
        order_by=order_by,
        descending=descending,
        where=where,
    )


def get_columns(
    *columns: ColumnSpec,
    limit: int | None = None,
    offset: int = 0,
    success_only: bool = False,
    as_dict: bool = True,
    order_by: ColumnSpec | None = None,
    descending: bool = False,
    where: Any = None,
) -> list[dict[str, Any]] | list[tuple[Any, ...]]:
    """Fetch multiple columns using default Stash instance."""
    return get_default_stash().get_columns(
        *columns,
        limit=limit,
        offset=offset,
        success_only=success_only,
        as_dict=as_dict,
        order_by=order_by,
        descending=descending,
        where=where,
    )


def get_map(
    key_column: ColumnSpec,
    value_column: ColumnSpec,
    *,
    limit: int | None = None,
    offset: int = 0,
    success_only: bool = False,
    where: Any = None,
) -> dict[Any, Any]:
    """Extract key-value mapping dictionary using default Stash instance."""
    return get_default_stash().get_map(
        key_column,
        value_column,
        limit=limit,
        offset=offset,
        success_only=success_only,
        where=where,
    )


def get_documents(
    columns: Sequence[ColumnSpec] | ColumnSpec | None = None,
    *,
    limit: int = 100,
    offset: int = 0,
    success_only: bool = False,
    status_code: int | None = None,
    url_contains: str | None = None,
    order_by: ColumnSpec = "id",
    descending: bool = False,
    as_dict: bool = False,
) -> list[Document] | list[dict[str, Any]]:
    """Query multiple documents using default Stash instance."""
    return get_default_stash().get_documents(
        columns=columns,
        limit=limit,
        offset=offset,
        success_only=success_only,
        status_code=status_code,
        url_contains=url_contains,
        order_by=order_by,
        descending=descending,
        as_dict=as_dict,
    )


def iter_documents(
    columns: Sequence[ColumnSpec] | ColumnSpec | None = None,
    *,
    batch_size: int = 500,
    success_only: bool = False,
    status_code: int | None = None,
    order_by: ColumnSpec = "id",
    descending: bool = False,
) -> Iterator[Document]:
    """Stream documents using default Stash instance."""
    return get_default_stash().iter_documents(
        columns=columns,
        batch_size=batch_size,
        success_only=success_only,
        status_code=status_code,
        order_by=order_by,
        descending=descending,
    )


def iter_records(
    columns: Sequence[ColumnSpec] | ColumnSpec,
    *,
    batch_size: int = 500,
    success_only: bool = False,
    status_code: int | None = None,
    order_by: ColumnSpec = "id",
    descending: bool = False,
) -> Iterator[dict[str, Any]]:
    """Stream row dictionaries using default Stash instance."""
    return get_default_stash().iter_records(
        columns=columns,
        batch_size=batch_size,
        success_only=success_only,
        status_code=status_code,
        order_by=order_by,
        descending=descending,
    )


def get_markdown(id_or_url: int | str) -> str | None:
    """Fetch markdown content for document using default Stash instance."""
    return get_default_stash().get_markdown(id_or_url)


def get_html(id_or_url: int | str) -> str | None:
    """Fetch HTML content for document using default Stash instance."""
    return get_default_stash().get_html(id_or_url)


def get_metadata(id_or_url: int | str) -> dict[str, Any] | None:
    """Fetch metadata dictionary for document using default Stash instance."""
    return get_default_stash().get_metadata(id_or_url)


def count(
    *,
    success_only: bool = False,
    url_contains: str | None = None,
    status_code: int | None = None,
) -> int:
    """Count documents using default Stash instance."""
    return get_default_stash().count(
        success_only=success_only,
        url_contains=url_contains,
        status_code=status_code,
    )


def exists(id_or_url: int | str) -> bool:
    """Check if document exists using default Stash instance."""
    return get_default_stash().exists(id_or_url)


def list_columns() -> list[str]:
    """List available columns using default Stash instance."""
    return get_default_stash().list_columns()


def search_similar(
    query_text: str,
    top_k: int = 10,
    hybrid_blend: float = 0.5,
    embed_url: str | None = None,
    as_search_results: bool = False,
) -> list[dict[str, Any]] | list[Any]:
    """Perform hybrid vector and keyword search using default Stash instance."""
    return get_default_stash().search_hybrid(
        query_text,
        top_k=top_k,
        hybrid_blend=hybrid_blend,
        embed_url=embed_url,
        as_search_results=as_search_results,
    )


def search_hybrid(
    query_text: str,
    top_k: int = 10,
    hybrid_blend: float = 0.5,
    embed_url: str | None = None,
    as_search_results: bool = False,
) -> list[dict[str, Any]] | list[Any]:
    """Perform hybrid search combining vector, keyword, and zerank-2 rerank."""
    return get_default_stash().search_hybrid(
        query_text,
        top_k=top_k,
        hybrid_blend=hybrid_blend,
        embed_url=embed_url,
        as_search_results=as_search_results,
    )


def search_keyword(
    query_text: str,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Perform keyword BM25 search using default Stash instance."""
    return get_default_stash().search_keyword(query_text, top_k=top_k)


def score_vector(
    vector: Sequence[float],
    top_k: int = 10,
    as_search_results: bool = False,
) -> list[dict[str, Any]] | list[Any]:
    """Score documents against an arbitrary vector using default Stash instance."""
    return get_default_stash().score_vector(
        vector, top_k=top_k, as_search_results=as_search_results
    )


def get_document_detail(
    doc_id: int,
    clean: bool = True,
) -> dict[str, Any] | None:
    """Fetch full document content and metadata by doc_id using default Stash instance."""
    return get_default_stash().get_document_detail(doc_id=doc_id, clean=clean)


def get_documents_detail(
    doc_ids: Sequence[int],
    clean: bool = True,
) -> list[dict[str, Any]]:
    """Fetch full document content and metadata for multiple doc_ids using default Stash instance."""
    return get_default_stash().get_documents_detail(doc_ids=doc_ids, clean=clean)


