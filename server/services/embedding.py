"""Runtime vector embedding inference and scoring using sqlite-vec."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from typing import Any, Sequence

import httpx
import numpy as np
import sqlite_vec

from server.config import (
    CHUNK_FANOUT,
    DEFAULT_DB_PATH,
    DEFAULT_EMBED_API_KEY,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_URL,
    DEFAULT_USEARCH_PATH,
    MAX_EMBED_CHARS,
    MAX_RETRIES,
    QUERY_PREFIX,
)
from server.models.text import clean_normal_text, clean_passage_snippet, extract_title_from_text

logger = logging.getLogger(__name__)

_usearch_index_cache: dict[str, Any] = {}
_usearch_lock = threading.Lock()


def get_usearch_index(index_path: str | None = None) -> Any | None:
    """Retrieve or memory-map the USearch HNSW vector index (thread-safe singleton)."""
    path = index_path or DEFAULT_USEARCH_PATH
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with _usearch_lock:
        if path in _usearch_index_cache:
            return _usearch_index_cache[path]
        try:
            from usearch.index import Index, MetricKind, ScalarKind
            index = Index(ndim=768, metric=MetricKind.Cos, dtype=ScalarKind.F16)
            index.view(path)
            _usearch_index_cache[path] = index
            logger.info("Loaded USearch index view from %s (%d keys)", path, len(index))
            return index
        except Exception as e:
            logger.warning("Could not load USearch index from %s: %s", path, e)
            return None

REGEX_HORIZONTAL_WS = re.compile(r"[ \t]+")
REGEX_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def prepare_embedding_text(text: str, max_chars: int = MAX_EMBED_CHARS) -> str:
    """Prepare and truncate markdown text for embedding model."""
    if not text:
        return ""

    cleaned = REGEX_HORIZONTAL_WS.sub(" ", text.strip())
    cleaned = REGEX_EXCESS_NEWLINES.sub("\n\n", cleaned)

    if len(cleaned) <= max_chars:
        return cleaned

    truncated = cleaned[:max_chars]
    last_space = max(truncated.rfind(" "), truncated.rfind("\n"))
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]

    return truncated.strip()


class EmbeddingClient:
    """Persistent HTTP client for OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        endpoint_url: str = DEFAULT_EMBED_URL,
        model_name: str = DEFAULT_EMBED_MODEL,
        timeout: float = 120.0,
        api_key: str = DEFAULT_EMBED_API_KEY,
        max_chars: int = MAX_EMBED_CHARS,
    ):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.timeout = timeout
        self.api_key = api_key
        self.max_chars = max_chars
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            timeout=timeout,
            headers=headers,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=120.0),
        )

    def _embed_batch_direct(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self.model_name,
            "input": list(texts),
        }

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.post(self.endpoint_url, json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"retryable status {resp.status_code}: {resp.text[:200]}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                sorted_items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_items]
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last_error = e
                backoff = min(30.0, 1.5 * (2**attempt))
                logger.warning(
                    "Embedding request failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError(f"Embedding request failed after {MAX_RETRIES} attempts: {last_error}")

    def embed_single(self, text: str) -> list[float]:
        """Embed a single string into a float vector with adaptive length truncation."""
        limits = (self.max_chars, 500, 300, 150)
        for limit in dict.fromkeys(limits):
            clean = prepare_embedding_text(text, max_chars=limit) or "empty"
            try:
                res = self._embed_batch_direct([clean])
                if res:
                    return res[0]
            except Exception as e:
                logger.warning("Embedding failed at %d chars: %s", limit, e)
                time.sleep(0.2)
        return self._embed_batch_direct(["empty"])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of strings in a single HTTP request."""
        if not texts:
            return []

        clean_inputs = [prepare_embedding_text(t, max_chars=self.max_chars) or "empty" for t in texts]

        try:
            return self._embed_batch_direct(clean_inputs)
        except Exception as e:
            logger.warning("Batch embedding failed (%s); falling back to resilient individual embeddings...", e)
            results = []
            for t in texts:
                results.append(self.embed_single(t))
            return results


_cached_embed_clients: dict[tuple[str, str, str], EmbeddingClient] = {}


def get_shared_embed_client(
    endpoint_url: str = DEFAULT_EMBED_URL,
    model_name: str = DEFAULT_EMBED_MODEL,
    api_key: str = DEFAULT_EMBED_API_KEY,
) -> EmbeddingClient:
    """Retrieve or create a persistent keep-alive EmbeddingClient."""
    # Keyless for local endpoints
    if "127.0.0.1" in endpoint_url or "localhost" in endpoint_url:
        api_key = ""
    key = (endpoint_url, model_name, api_key)
    client = _cached_embed_clients.get(key)
    if client is None or client._client.is_closed:
        client = EmbeddingClient(endpoint_url=endpoint_url, model_name=model_name, api_key=api_key)
        _cached_embed_clients[key] = client
    return client


def _score_chunks(
    conn: sqlite3.Connection,
    serialized_vec: bytes,
    top_k: int,
    include_rerank_text: bool,
) -> list[dict[str, Any]]:
    """kNN over chunk vectors, keeping each document's best-matching chunk, resolving parent context, and summary."""
    chunk_k = max(top_k * CHUNK_FANOUT, top_k)
    query = """
    SELECT
        v.chunk_id,
        v.distance,
        c.doc_id,
        c.chunk_text,
        c.chunk_index,
        s.url,
        COALESCE(json_extract(s.metadata, '$.title'), '') AS title,
        COALESCE(ds.summary, '') AS summary,
        COALESCE(ds.published_date, substr(s.saved_at, 1, 10)) AS doc_date,
        s.saved_at,
        COALESCE(p.chunk_text, c.chunk_text) AS parent_text
    FROM vec_chunks v
    JOIN document_chunks c ON c.chunk_id = v.chunk_id
    LEFT JOIN document_chunks p ON p.chunk_id = c.parent_chunk_id
    JOIN documents s ON s.id = c.doc_id
    LEFT JOIN document_summaries ds ON ds.doc_id = c.doc_id
    WHERE v.embedding MATCH ? AND k = ?
    ORDER BY v.distance ASC
    """
    rows = conn.execute(query, (serialized_vec, chunk_k)).fetchall()

    best: dict[int, dict[str, Any]] = {}
    context: dict[int, dict[str, float]] = {}  # doc_id -> {passage_text: best_distance}
    for _chunk_id, distance, doc_id, chunk_text, _chunk_index, url, title, summary, doc_date, saved_at, parent_text in rows:
        distance = float(distance)
        effective_passage = (parent_text or chunk_text or "").strip()
        if effective_passage:
            curr_passages = context.setdefault(doc_id, {})
            if effective_passage not in curr_passages or distance < curr_passages[effective_passage]:
                curr_passages[effective_passage] = distance

        if doc_id in best and best[doc_id]["distance"] <= distance:
            continue
        text = effective_passage or ""
        clean_snip = clean_passage_snippet(text, max_chars=900)
        resolved_title = (title or "").strip() or extract_title_from_text(text, url)
        best[doc_id] = {
            "doc_id": doc_id,
            "distance": distance,
            "score": 1.0 - distance,
            "url": url,
            "title": resolved_title,
            "snippet": clean_snip,
            "summary": summary or "",
            "published_date": str(doc_date) if doc_date else "",
            "saved_at": str(saved_at) if saved_at else "",
        }

    results = sorted(best.values(), key=lambda r: r["distance"])[:top_k]
    if include_rerank_text:
        for result in results:
            doc_passages = context.get(result["doc_id"], {})
            sorted_passages = sorted(doc_passages.items(), key=lambda item: item[1])
            passages = [p for p, _ in sorted_passages[:3]]
            result["rerank_text"] = "\n\n".join(passages)[:8000]
    return results


def _score_chunks_usearch(
    conn: sqlite3.Connection,
    vector: Sequence[float],
    top_k: int,
    include_rerank_text: bool,
    index: Any,
) -> list[dict[str, Any]]:
    """Ultra-fast kNN over USearch HNSW index, resolving chunk/parent context via indexed SQLite query."""
    chunk_k = max(top_k * CHUNK_FANOUT, top_k)
    q_vec = np.asarray(vector, dtype=np.float32)
    matches = index.search(q_vec, chunk_k)

    if len(matches) == 0:
        return []

    chunk_ids = [int(k) for k in matches.keys]
    distances = {int(k): float(d) for k, d in zip(matches.keys, matches.distances)}

    placeholders = ",".join("?" for _ in chunk_ids)
    query = f"""
    SELECT
        c.chunk_id,
        c.doc_id,
        c.chunk_text,
        c.chunk_index,
        s.url,
        COALESCE(json_extract(s.metadata, '$.title'), '') AS title,
        COALESCE(ds.summary, '') AS summary,
        COALESCE(ds.published_date, substr(s.saved_at, 1, 10)) AS doc_date,
        s.saved_at,
        COALESCE(p.chunk_text, c.chunk_text) AS parent_text
    FROM document_chunks c
    LEFT JOIN document_chunks p ON p.chunk_id = c.parent_chunk_id
    JOIN documents s ON s.id = c.doc_id
    LEFT JOIN document_summaries ds ON ds.doc_id = c.doc_id
    WHERE c.chunk_id IN ({placeholders})
    """
    rows = conn.execute(query, chunk_ids).fetchall()

    best: dict[int, dict[str, Any]] = {}
    context: dict[int, dict[str, float]] = {}

    for chunk_id, doc_id, chunk_text, _chunk_index, url, title, summary, doc_date, saved_at, parent_text in rows:
        distance = distances.get(chunk_id, 1.0)
        effective_passage = (parent_text or chunk_text or "").strip()
        if effective_passage:
            curr_passages = context.setdefault(doc_id, {})
            if effective_passage not in curr_passages or distance < curr_passages[effective_passage]:
                curr_passages[effective_passage] = distance

        if doc_id in best and best[doc_id]["distance"] <= distance:
            continue
        text = effective_passage or ""
        clean_snip = clean_passage_snippet(text, max_chars=900)
        resolved_title = (title or "").strip() or extract_title_from_text(text, url)
        best[doc_id] = {
            "doc_id": doc_id,
            "distance": distance,
            "score": 1.0 - distance,
            "url": url,
            "title": resolved_title,
            "snippet": clean_snip,
            "summary": summary or "",
            "published_date": str(doc_date) if doc_date else "",
            "saved_at": str(saved_at) if saved_at else "",
        }

    results = sorted(best.values(), key=lambda r: r["distance"])[:top_k]
    if include_rerank_text:
        for result in results:
            doc_passages = context.get(result["doc_id"], {})
            sorted_passages = sorted(doc_passages.items(), key=lambda item: item[1])
            passages = [p for p, _ in sorted_passages[:3]]
            result["rerank_text"] = "\n\n".join(passages)[:8000]
    return results


def _score_documents(
    conn: sqlite3.Connection,
    serialized_vec: bytes,
    top_k: int,
    include_rerank_text: bool,
) -> list[dict[str, Any]]:
    """Doc-level kNN over vec_documents (one truncated vector per page)."""
    cursor = conn.cursor()

    query = """
    SELECT
        v.doc_id,
        v.distance,
        (1.0 - v.distance) AS score,
        s.url,
        COALESCE(json_extract(s.metadata, '$.title'), '') AS title,
        substr(s.markdown, 1, 300) AS snippet{rerank_text_column},
        s.saved_at
    FROM vec_documents v
    JOIN documents s ON s.id = v.doc_id
    WHERE v.embedding MATCH ? AND k = ?
    ORDER BY v.distance ASC
    """.format(
        rerank_text_column=", substr(COALESCE(s.markdown, s.html, ''), 1, 8000) AS rerank_text"
        if include_rerank_text
        else ""
    )

    cursor.execute(query, (serialized_vec, top_k))
    rows = cursor.fetchall()

    results = []
    for row in rows:
        doc_id, distance, score, url, title, snippet = row[:6]
        saved_at = row[-1]
        result = {
            "doc_id": doc_id,
            "distance": float(distance),
            "score": float(score),
            "url": url,
            "title": title,
            "snippet": clean_normal_text(snippet)[:300].strip() if snippet else "",
            "saved_at": str(saved_at) if saved_at else "",
        }
        if include_rerank_text:
            result["rerank_text"] = row[6] or ""
        results.append(result)

    return results


def score_vector(
    vector: Sequence[float],
    top_k: int = 10,
    db_path: str = DEFAULT_DB_PATH,
    include_rerank_text: bool = False,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Score documents against a given float vector using USearch (fast) or sqlite-vec (fallback)."""
    should_close = False
    if conn is None:
        conn = sqlite3.connect(db_path, timeout=60.0)
        should_close = True

    try:
        has_chunks = False
        try:
            row = conn.execute("SELECT EXISTS(SELECT 1 FROM document_chunks)").fetchone()
            if row and row[0]:
                has_chunks = True
        except Exception:
            pass

        # Try ultra-fast USearch index first
        usearch_idx = get_usearch_index()
        if usearch_idx is not None and has_chunks:
            return _score_chunks_usearch(conn, vector, top_k, include_rerank_text, usearch_idx)

        # Fallback to sqlite-vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        serialized_vec = sqlite_vec.serialize_float32(list(vector))

        if has_chunks:
            results = _score_chunks(conn, serialized_vec, top_k, include_rerank_text)
        else:
            results = _score_documents(conn, serialized_vec, top_k, include_rerank_text)
        return results
    finally:
        if should_close:
            conn.close()


def search_similar(
    query_text: str,
    top_k: int = 10,
    db_path: str = DEFAULT_DB_PATH,
    embed_url: str = DEFAULT_EMBED_URL,
    model_name: str = DEFAULT_EMBED_MODEL,
    api_key: str = DEFAULT_EMBED_API_KEY,
    include_rerank_text: bool = False,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """End-to-end vector search using persistent keep-alive client and sqlite-vec."""
    client = get_shared_embed_client(endpoint_url=embed_url, model_name=model_name, api_key=api_key)
    q_vec = client.embed_single(QUERY_PREFIX + query_text)

    return score_vector(
        q_vec,
        top_k=top_k,
        db_path=db_path,
        include_rerank_text=include_rerank_text,
        conn=conn,
    )
