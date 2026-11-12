"""
embed_stash.py - Vector Embedding Pipeline & Scoring with sqlite-vec.

Embeds documents (their markdown version) from stash.db using the EmbeddingGemma model
served by the FTULabs gateway (FTULABS_API_ENDPOINT/v1/embeddings), and indexes &
scores vectors using sqlite-vec with cosine similarity.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import httpx
import sqlite_vec
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from server.config import (
        CHUNK_CHARS,
        CHUNK_FANOUT,
        CHUNK_OVERLAP,
        DEFAULT_CHUNK_BATCH_SIZE,
        DEFAULT_CONCURRENCY,
        DEFAULT_DB_PATH,
        DEFAULT_EMBED_API_KEY,
        DEFAULT_EMBED_BATCH_SIZE,
        DEFAULT_EMBED_MODEL,
        DEFAULT_EMBED_URL,
        DOC_PREFIX,
        MAX_EMBED_CHARS,
        MAX_RETRIES,
        QUERY_PREFIX,
    )
    DEFAULT_MODEL = DEFAULT_EMBED_MODEL
    DEFAULT_API_KEY = DEFAULT_EMBED_API_KEY
    DEFAULT_BATCH_SIZE = DEFAULT_EMBED_BATCH_SIZE
except ImportError:
    DEFAULT_DB_PATH = "stash.db"
    FTULABS_API_ENDPOINT = os.environ.get("FTULABS_API_ENDPOINT", "").rstrip("/")
    DEFAULT_EMBED_URL = os.environ.get(
        "EMBED_URL",
        f"{FTULABS_API_ENDPOINT}/embeddings" if FTULABS_API_ENDPOINT else "http://ml:8085/v1/embeddings",
    )
    DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "unsloth/embeddinggemma-2-GGUF")
    DEFAULT_API_KEY = os.environ.get("EMBED_API_KEY") or os.environ.get("FTULABS_API_KEY", "")
    DEFAULT_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
    DEFAULT_CHUNK_BATCH_SIZE = int(os.environ.get("EMBED_CHUNK_BATCH_SIZE", "16"))
    DEFAULT_CONCURRENCY = int(os.environ.get("EMBED_CONCURRENCY", "4"))
    MAX_RETRIES = int(os.environ.get("EMBED_MAX_RETRIES", "6"))
    CHUNK_CHARS = int(os.environ.get("EMBED_CHUNK_CHARS", "4000"))
    CHUNK_OVERLAP = int(os.environ.get("EMBED_CHUNK_OVERLAP", "400"))
    MAX_EMBED_CHARS = int(os.environ.get("EMBED_MAX_CHARS", str(CHUNK_CHARS + CHUNK_OVERLAP)))
    CHUNK_FANOUT = int(os.environ.get("EMBED_CHUNK_FANOUT", "5"))
    QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "task: search result | query: ")
    DOC_PREFIX = os.environ.get("EMBED_DOC_PREFIX", "title: none | text: ")

VECTOR_DIM = 768
MAX_CHAR_LENGTH = 700

REGEX_HORIZONTAL_WS = re.compile(r"[ \t]+")
REGEX_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def prepare_embedding_text(text: str, max_chars: int = MAX_CHAR_LENGTH) -> str:
    """
    Prepare and truncate markdown text for EmbeddingGemma.
    Collapses whitespace and slices at word boundary within the token budget.
    """
    if not text:
        return ""

    # Collapse excessive horizontal spaces and empty lines
    cleaned = REGEX_HORIZONTAL_WS.sub(" ", text.strip())
    cleaned = REGEX_EXCESS_NEWLINES.sub("\n\n", cleaned)

    if len(cleaned) <= max_chars:
        return cleaned

    # Truncate at word boundary
    truncated = cleaned[:max_chars]
    last_space = max(truncated.rfind(" "), truncated.rfind("\n"))
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]

    return truncated.strip()


def split_into_chunks(
    text: str,
    max_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split markdown into overlapping chunks of at most `max_chars` characters.

    Paragraph boundaries are respected; only a single paragraph longer than the budget is
    hard-split at a word boundary. Each chunk after the first is seeded with the tail of
    the previous one so sentences spanning a boundary are still represented.
    """
    if not text or not text.strip():
        return []

    cleaned = REGEX_HORIZONTAL_WS.sub(" ", text.strip())
    cleaned = REGEX_EXCESS_NEWLINES.sub("\n\n", cleaned)
    if len(cleaned) <= max_chars:
        return [cleaned]

    # First pass: break into units no larger than max_chars.
    units: list[str] = []
    for para in cleaned.split("\n\n"):
        para = para.strip()
        while len(para) > max_chars:
            cut = para.rfind(" ", 0, max_chars)
            if cut < max_chars // 2:
                cut = max_chars
            units.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            units.append(para)

    # Second pass: pack units into chunks, carrying an overlap tail forward.
    chunks: list[str] = []
    buf = ""
    for unit in units:
        candidate = f"{buf}\n\n{unit}" if buf else unit
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            if tail:
                space = tail.find(" ")
                tail = tail[space + 1:] if space != -1 else ""
            buf = f"{tail}\n\n{unit}" if tail else unit
        else:
            buf = unit
    if buf:
        chunks.append(buf)

    return chunks


class EmbeddingClient:
    """Client for an OpenAI-compatible embeddings API (FTULabs gateway / llama.cpp)."""

    def __init__(
        self,
        endpoint_url: str = DEFAULT_EMBED_URL,
        model_name: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        api_key: str = DEFAULT_API_KEY,
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
                # 4xx other than 429 means the request itself is wrong; do not retry.
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last_error = e
                backoff = min(30.0, 1.5 * (2**attempt))
                logger.warning(f"Embedding request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}; retrying in {backoff:.1f}s")
                time.sleep(backoff)

        raise RuntimeError(f"Embedding request failed after {MAX_RETRIES} attempts: {last_error}")

    def embed_single(self, text: str) -> list[float]:
        """Embed a single string into a 768-dim float vector with adaptive length truncation."""
        limits = (self.max_chars, 500, 300, 150)
        for limit in dict.fromkeys(limits):
            clean = prepare_embedding_text(text, max_chars=limit) or "empty"
            try:
                res = self._embed_batch_direct([clean])
                if res:
                    return res[0]
            except Exception as e:
                logger.warning(f"Embedding failed at {limit} chars: {e}")
                time.sleep(0.2)
        # Fallback to placeholder
        return self._embed_batch_direct(["empty"])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed a batch of strings in a single HTTP request.
        Falls back to resilient individual embeddings if the batch fails.
        """
        if not texts:
            return []

        clean_inputs = [prepare_embedding_text(t, max_chars=self.max_chars) or "empty" for t in texts]

        try:
            return self._embed_batch_direct(clean_inputs)
        except Exception as e:
            logger.warning(f"Batch embedding failed ({e}); falling back to resilient individual embeddings...")
            results = []
            for t in texts:
                results.append(self.embed_single(t))
            return results

    def close(self) -> None:
        self._client.close()


def init_sqlite_vec_tables(conn: sqlite3.Connection) -> None:
    """Initialize sqlite-vec extension and virtual tables."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
            doc_id integer primary key,
            embedding float[{VECTOR_DIM}] distance_metric=cosine
        )
        """
    )
    init_chunk_tables(conn)
    conn.commit()


def init_chunk_tables(conn: sqlite3.Connection) -> None:
    """Create the chunk text table and its sqlite-vec companion if missing."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            token_count INTEGER,
            FOREIGN KEY (doc_id) REFERENCES storedcrawlresult(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_doc ON document_chunks(doc_id)"
    )
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id integer primary key,
            embedding float[{VECTOR_DIM}] distance_metric=cosine
        )
        """
    )
    conn.commit()


def get_vec_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with sqlite-vec extension loaded, WAL mode, and memory-mapped IO."""
    conn = sqlite3.connect(db_path, timeout=60.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA mmap_size=2147483648")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA temp_store=MEMORY")
    init_sqlite_vec_tables(conn)
    return conn


def run_embedding_for_all_documents(
    db_path: str = DEFAULT_DB_PATH,
    embed_url: str = DEFAULT_EMBED_URL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
    skip_existing: bool = True,
    limit: int | None = None,
    verbose: bool = True,
    model_name: str = DEFAULT_MODEL,
    api_key: str = DEFAULT_API_KEY,
) -> dict[str, int]:
    """
    Generate embeddings for all document markdowns in stash.db and store in vec_documents.

    Args:
        db_path: Path to stash.db.
        embed_url: OpenAI-compatible embeddings endpoint.
        batch_size: Batch size per HTTP request.
        concurrency: Number of embedding requests kept in flight.
        skip_existing: If True, skips documents that already have vectors in vec_documents.
        limit: Optional maximum number of documents to embed.
        verbose: Display progress bar.
        model_name: Embedding model id sent in the request payload.
        api_key: Bearer token for the embeddings gateway.

    Returns:
        Statistics dictionary.
    """
    client = EmbeddingClient(
        endpoint_url=embed_url,
        model_name=model_name,
        api_key=api_key,
        max_chars=MAX_CHAR_LENGTH,
    )
    conn = get_vec_connection(db_path)
    cursor = conn.cursor()

    # Determine documents to process
    if skip_existing:
        cursor.execute(
            """
            SELECT s.id, s.markdown
            FROM storedcrawlresult s
            WHERE length(s.markdown) > 0
              AND s.id NOT IN (SELECT doc_id FROM vec_documents)
            ORDER BY s.id
            """
        )
    else:
        cursor.execute(
            """
            SELECT s.id, s.markdown
            FROM storedcrawlresult s
            WHERE length(s.markdown) > 0
            ORDER BY s.id
            """
        )

    rows = cursor.fetchall()
    if limit is not None:
        rows = rows[:limit]

    total_to_process = len(rows)

    if verbose:
        existing_count = cursor.execute("SELECT count(*) FROM vec_documents").fetchone()[0]
        print(f"Embedding stash documents with {embed_url}...")
        print(f"Found {existing_count} existing vectors, {total_to_process} documents to embed.")

    if total_to_process == 0:
        if verbose:
            print("All documents are already embedded!")
        conn.close()
        client.close()
        return {"total": total_to_process, "embedded": 0}

    pbar = tqdm(
        total=total_to_process,
        disable=not verbose,
        unit="doc",
        desc="Embedding",
        mininterval=1.0,
        smoothing=0.05,
        dynamic_ncols=True,
        file=sys.stderr,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
        ),
    )
    embedded_count = 0
    start_time = time.time()

    def flush_chunk(chunk: Sequence[tuple[int, str]], embeddings: list[list[float]]) -> None:
        """Write one batch of vectors. SQLite writes stay on the main thread."""
        nonlocal embedded_count
        doc_ids = [r[0] for r in chunk]

        # vec0 rejects INSERT OR REPLACE on an existing primary key, so stale vectors
        # must be deleted before re-inserting.
        insert_data = [
            (doc_id, sqlite_vec.serialize_float32(emb))
            for doc_id, emb in zip(doc_ids, embeddings)
        ]
        placeholders = ",".join("?" * len(doc_ids))
        conn.execute(f"DELETE FROM vec_documents WHERE doc_id IN ({placeholders})", doc_ids)
        conn.executemany(
            "INSERT INTO vec_documents(doc_id, embedding) VALUES (?, ?)",
            insert_data,
        )
        conn.commit()

        embedded_count += len(chunk)
        elapsed_now = time.time() - start_time
        cur_rate = embedded_count / elapsed_now if elapsed_now > 0 else 0
        pbar.update(len(chunk))
        pbar.set_postfix_str(f"batch={len(chunk)} inflight={concurrency} avg={cur_rate:.1f} doc/s", refresh=False)

    chunks = [rows[i : i + batch_size] for i in range(0, total_to_process, batch_size)]

    try:
        if concurrency <= 1:
            for chunk in chunks:
                flush_chunk(chunk, client.embed_batch([DOC_PREFIX + r[1] for r in chunk]))
        else:
            # Keep `concurrency` requests in flight: the gateway's GPU is the bottleneck and
            # single-request throughput saturates well before its slot count does.
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                pending: deque[tuple[Sequence[tuple[int, str]], Any]] = deque()
                for chunk in chunks:
                    pending.append(
                        (chunk, pool.submit(client.embed_batch, [DOC_PREFIX + r[1] for r in chunk]))
                    )
                    while len(pending) > concurrency:
                        done_chunk, done_future = pending.popleft()
                        flush_chunk(done_chunk, done_future.result())
                while pending:
                    done_chunk, done_future = pending.popleft()
                    flush_chunk(done_chunk, done_future.result())

    finally:
        pbar.close()
        conn.close()
        client.close()

    elapsed = time.time() - start_time
    rate = embedded_count / elapsed if elapsed > 0 else 0
    if verbose:
        print(f"Finished embedding {embedded_count} documents in {elapsed:.2f}s ({rate:.1f} doc/s).")

    return {"total": total_to_process, "embedded": embedded_count}


def run_chunk_embedding(
    db_path: str = DEFAULT_DB_PATH,
    embed_url: str = DEFAULT_EMBED_URL,
    batch_size: int = DEFAULT_CHUNK_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
    skip_existing: bool = True,
    limit: int | None = None,
    verbose: bool = True,
    model_name: str = DEFAULT_MODEL,
    api_key: str = DEFAULT_API_KEY,
    chunk_chars: int = CHUNK_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict[str, int]:
    """
    Split every document into overlapping chunks and embed each chunk into vec_chunks.

    Unlike the doc-level path (one truncated vector per page), this indexes the whole
    document, so retrieval can match a passage deep inside a long page.

    Args:
        skip_existing: If True, skips documents that already have rows in document_chunks.
        chunk_chars: Maximum characters per chunk (~4 chars/token).
        chunk_overlap: Characters of the previous chunk carried into the next one.

    Returns:
        Statistics dictionary with document and chunk counts.
    """
    client = EmbeddingClient(
        endpoint_url=embed_url,
        model_name=model_name,
        api_key=api_key,
        max_chars=chunk_chars + chunk_overlap + len(DOC_PREFIX),
    )
    conn = get_vec_connection(db_path)
    conn.execute("PRAGMA synchronous=NORMAL")
    cursor = conn.cursor()

    if skip_existing:
        cursor.execute(
            """
            SELECT s.id, s.markdown
            FROM storedcrawlresult s
            WHERE length(s.markdown) > 0
              AND s.id NOT IN (SELECT DISTINCT doc_id FROM document_chunks)
            ORDER BY s.id
            """
        )
    else:
        cursor.execute(
            """
            SELECT s.id, s.markdown
            FROM storedcrawlresult s
            WHERE length(s.markdown) > 0
            ORDER BY s.id
            """
        )

    rows = cursor.fetchall()
    if limit is not None:
        rows = rows[:limit]
    total_docs = len(rows)

    if verbose:
        existing_chunks = cursor.execute("SELECT count(*) FROM document_chunks").fetchone()[0]
        print(f"Chunk-embedding {total_docs} documents via {embed_url}...")
        print(
            f"{existing_chunks} existing chunks | chunk={chunk_chars} chars "
            f"(~{chunk_chars // 4} tok), overlap={chunk_overlap}, "
            f"batch={batch_size}, concurrency={concurrency}"
        )

    if total_docs == 0:
        if verbose:
            print("All documents are already chunked!")
        conn.close()
        client.close()
        return {"docs": 0, "chunks": 0}

    def delete_doc_chunks(doc_id: int) -> None:
        stale = [r[0] for r in conn.execute(
            "SELECT chunk_id FROM document_chunks WHERE doc_id = ?", (doc_id,)
        )]
        if stale:
            placeholders = ",".join("?" * len(stale))
            conn.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})", stale)
            conn.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))

    def iter_batches():
        """Yield (items, docs_in_batch) where items are (chunk_id, text) pairs."""
        buf: list[tuple[int, str]] = []
        docs_in_buf = 0
        for doc_id, markdown in rows:
            if not skip_existing:
                delete_doc_chunks(doc_id)
            for index, text in enumerate(split_into_chunks(markdown, chunk_chars, chunk_overlap)):
                cur = conn.execute(
                    "INSERT INTO document_chunks(doc_id, chunk_index, chunk_text, token_count)"
                    " VALUES (?, ?, ?, ?)",
                    (doc_id, index, text, len(text) // 4),
                )
                buf.append((cur.lastrowid, text))
                if len(buf) >= batch_size:
                    yield buf, docs_in_buf
                    buf, docs_in_buf = [], 0
            docs_in_buf += 1
        if buf or docs_in_buf:
            yield buf, docs_in_buf

    pbar = tqdm(
        total=total_docs,
        disable=not verbose,
        unit="doc",
        desc="Chunking",
        mininterval=1.0,
        smoothing=0.05,
        dynamic_ncols=True,
        file=sys.stderr,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} docs "
            "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
        ),
    )
    start_time = time.time()
    chunk_count = 0

    def write_chunk_vectors(
        items: Sequence[tuple[int, str]],
        embeddings: list[list[float]],
        docs_done: int,
    ) -> None:
        nonlocal chunk_count
        if items:
            chunk_ids = [cid for cid, _ in items]
            insert_data = [
                (cid, sqlite_vec.serialize_float32(emb))
                for cid, emb in zip(chunk_ids, embeddings)
            ]
            placeholders = ",".join("?" * len(chunk_ids))
            conn.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
            conn.executemany(
                "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                insert_data,
            )
            chunk_count += len(items)
        conn.commit()

        pbar.update(docs_done)
        elapsed_now = time.time() - start_time
        rate = chunk_count / elapsed_now if elapsed_now > 0 else 0
        pbar.set_postfix_str(f"chunks={chunk_count} {rate:.1f} chunk/s", refresh=False)

    try:
        if concurrency <= 1:
            for items, docs_done in iter_batches():
                write_chunk_vectors(
                    items,
                    client.embed_batch([DOC_PREFIX + t for _, t in items]),
                    docs_done,
                )
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                pending: deque[tuple[Sequence[tuple[int, str]], Any, int]] = deque()
                for items, docs_done in iter_batches():
                    pending.append(
                        (
                            items,
                            pool.submit(client.embed_batch, [DOC_PREFIX + t for _, t in items]),
                            docs_done,
                        )
                    )
                    while len(pending) > concurrency:
                        done_items, done_future, docs_in_batch = pending.popleft()
                        write_chunk_vectors(done_items, done_future.result(), docs_in_batch)
                while pending:
                    done_items, done_future, docs_in_batch = pending.popleft()
                    write_chunk_vectors(done_items, done_future.result(), docs_in_batch)

    finally:
        pbar.close()
        conn.close()
        client.close()

    elapsed = time.time() - start_time
    rate = chunk_count / elapsed if elapsed > 0 else 0
    if verbose:
        print(
            f"Finished chunk-embedding {total_docs} documents into {chunk_count} chunks "
            f"in {elapsed:.2f}s ({rate:.1f} chunk/s)."
        )

    return {"docs": total_docs, "chunks": chunk_count}


_cached_embed_clients: dict[tuple[str, str, str], EmbeddingClient] = {}


def get_shared_embed_client(
    endpoint_url: str = DEFAULT_EMBED_URL,
    model_name: str = DEFAULT_MODEL,
    api_key: str = DEFAULT_API_KEY,
) -> EmbeddingClient:
    """Retrieve or create a persistent keep-alive EmbeddingClient."""
    key = (endpoint_url, model_name, api_key)
    client = _cached_embed_clients.get(key)
    if client is None or client._client.is_closed:
        client = EmbeddingClient(endpoint_url=endpoint_url, model_name=model_name, api_key=api_key)
        _cached_embed_clients[key] = client
    return client


def score_vector(
    vector: Sequence[float],
    top_k: int = 10,
    db_path: str = DEFAULT_DB_PATH,
    include_rerank_text: bool = False,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """
    Score documents against a given 768-dim vector using sqlite-vec.

    Retrieval runs over chunk vectors when the corpus has been chunk-indexed, and falls
    back to the doc-level vec_documents index otherwise.
    """
    should_close = False
    if conn is None:
        conn = get_vec_connection(db_path)
        should_close = True

    try:
        serialized_vec = sqlite_vec.serialize_float32(list(vector))
        has_chunks = conn.execute(
            "SELECT EXISTS(SELECT 1 FROM document_chunks)"
        ).fetchone()[0]
        if has_chunks:
            results = _score_chunks(conn, serialized_vec, top_k, include_rerank_text)
        else:
            results = _score_documents(conn, serialized_vec, top_k, include_rerank_text)
        return results
    finally:
        if should_close:
            conn.close()


def _score_chunks(
    conn: sqlite3.Connection,
    serialized_vec: bytes,
    top_k: int,
    include_rerank_text: bool,
) -> list[dict[str, Any]]:
    """kNN over chunk vectors, keeping each document's best-matching chunk and summary."""
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
        s.saved_at
    FROM vec_chunks v
    JOIN document_chunks c ON c.chunk_id = v.chunk_id
    JOIN storedcrawlresult s ON s.id = c.doc_id
    LEFT JOIN document_summaries ds ON ds.doc_id = c.doc_id
    WHERE v.embedding MATCH ? AND k = ?
    ORDER BY v.distance ASC
    """
    rows = conn.execute(query, (serialized_vec, chunk_k)).fetchall()

    best: dict[int, dict[str, Any]] = {}
    context: dict[int, list[tuple[float, str]]] = {}
    for _chunk_id, distance, doc_id, chunk_text, _chunk_index, url, title, summary, doc_date, saved_at in rows:
        distance = float(distance)
        context.setdefault(doc_id, []).append((distance, chunk_text or ""))
        if doc_id in best and best[doc_id]["distance"] <= distance:
            continue
        text = chunk_text or ""
        try:
            from server.models import clean_normal_text
        except ImportError:
            from models import clean_normal_text
        clean_snip = clean_normal_text(text)
        best[doc_id] = {
            "doc_id": doc_id,
            "distance": distance,
            "score": 1.0 - distance,
            "url": url,
            "title": title,
            "snippet": clean_snip[:300].strip(),
            "summary": summary or "",
            "published_date": str(doc_date) if doc_date else "",
            "saved_at": str(saved_at) if saved_at else "",
        }

    results = sorted(best.values(), key=lambda r: r["distance"])[:top_k]
    if include_rerank_text:
        for result in results:
            passages = [t for _, t in sorted(context[result["doc_id"]])[:3]]
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
    JOIN storedcrawlresult s ON s.id = v.doc_id
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


def search_similar(
    query_text: str,
    top_k: int = 10,
    db_path: str = DEFAULT_DB_PATH,
    embed_url: str = DEFAULT_EMBED_URL,
    model_name: str = DEFAULT_MODEL,
    api_key: str = DEFAULT_API_KEY,
    include_rerank_text: bool = False,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """
    End-to-end vector search:
    1. Embeds query_text with persistent keep-alive embeddings client.
    2. Performs vector similarity search with sqlite-vec (reusing connection if provided).
    """
    client = get_shared_embed_client(endpoint_url=embed_url, model_name=model_name, api_key=api_key)
    q_vec = client.embed_single(QUERY_PREFIX + query_text)

    return score_vector(
        q_vec,
        top_k=top_k,
        db_path=db_path,
        include_rerank_text=include_rerank_text,
        conn=conn,
    )


def main() -> None:
    """CLI interface for embedding documents and searching vectors."""
    parser = argparse.ArgumentParser(
        description="Embed markdown documents into sqlite-vec and score vector similarities."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--embed-url",
        default=DEFAULT_EMBED_URL,
        help=f"OpenAI-compatible embeddings endpoint (default: {DEFAULT_EMBED_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Texts per embedding request (default: {DEFAULT_CHUNK_BATCH_SIZE} in chunks mode, "
        f"{DEFAULT_BATCH_SIZE} in docs mode)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"In-flight embedding requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for embeddings endpoint (default: env EMBED_API_KEY / FTULABS_API_KEY)",
    )
    parser.add_argument(
        "--mode",
        choices=("chunks", "docs"),
        default="chunks",
        help="chunks: index every passage into vec_chunks (default); "
        "docs: one truncated vector per document into vec_documents",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=CHUNK_CHARS,
        help=f"Characters per chunk in chunks mode (default: {CHUNK_CHARS} ~= {CHUNK_CHARS // 4} tokens)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Overlap carried between chunks (default: {CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--reindex-all",
        action="store_true",
        help="Reindex all documents instead of skipping already embedded ones",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N documents (useful for smoke tests)",
    )
    parser.add_argument(
        "--search",
        type=str,
        default="",
        help="Test search query against indexed vectors",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results for search (default: 5)",
    )

    args = parser.parse_args()

    if args.search:
        print(f"\nSearching for: {args.search!r} (top {args.top_k} results)...")
        hits = search_similar(
            args.search,
            top_k=args.top_k,
            db_path=args.db,
            embed_url=args.embed_url,
            model_name=args.model,
            api_key=args.api_key,
        )
        for i, hit in enumerate(hits, start=1):
            print(f"\n[{i}] Doc ID: {hit['doc_id']} | Score: {hit['score']:.4f} (distance: {hit['distance']:.4f})")
            print(f"    URL: {hit['url']}")
            print(f"    Snippet: {hit['snippet'][:120]}...")
    elif args.mode == "chunks":
        run_chunk_embedding(
            db_path=args.db,
            embed_url=args.embed_url,
            batch_size=args.batch_size or DEFAULT_CHUNK_BATCH_SIZE,
            concurrency=args.concurrency,
            skip_existing=not args.reindex_all,
            limit=args.limit,
            verbose=True,
            model_name=args.model,
            api_key=args.api_key,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
        )
    else:
        run_embedding_for_all_documents(
            db_path=args.db,
            embed_url=args.embed_url,
            batch_size=args.batch_size or DEFAULT_BATCH_SIZE,
            concurrency=args.concurrency,
            skip_existing=not args.reindex_all,
            limit=args.limit,
            verbose=True,
            model_name=args.model,
            api_key=args.api_key,
        )


if __name__ == "__main__":
    main()
