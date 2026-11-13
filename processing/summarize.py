"""
summarize_stash.py - Unified Document Summarization, Date Extraction, and Summary Vector Indexing.

Implements a 3-step production pipeline:
  1. Instant Programmatic Pre-Pass: Scans HTML microdata, OpenGraph, JSON-LD, and regex bylines.
  2. Single Unified LLM Pass: Uses a local or remote LLM (e.g., Gemma 4 E2B, Gemma 3 4B)
     to verify/extract the publication date and generate a high-quality 2-3 sentence summary in a single prompt.
  3. Summary Vector Embedding: Indexes the generated summaries into `vec_summaries` using sqlite-vec
     for dual-tier search (Document Summaries + Passage Chunks).
"""

from __future__ import annotations

import argparse
import json
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

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Environment loader ---
def load_env_file(path: Path | None = None) -> None:
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


load_env_file()

try:
    from server.config import (
        DEFAULT_DB_PATH,
        DEFAULT_EMBED_API_KEY,
        DEFAULT_EMBED_MODEL,
        DEFAULT_EMBED_URL,
        DEFAULT_LLM_API_KEY,
        DEFAULT_LLM_MODEL,
        DEFAULT_LLM_URL,
        DOC_PREFIX,
        MAX_PROMPT_CHARS,
        MAX_RETRIES,
        QUERY_PREFIX,
    )
except ImportError:
    DEFAULT_DB_PATH = "stash.db"
    DEFAULT_LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8888/v1/chat/completions")
    DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "unsloth/gemma-4-E2B-it-qat-GGUF")
    DEFAULT_LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
    DEFAULT_EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:8888/v1/embeddings")
    DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "unsloth/embeddinggemma-2-GGUF")
    DEFAULT_EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")
    DOC_PREFIX = os.environ.get("EMBED_DOC_PREFIX", "title: none | text: ")
    QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "task: search result | query: ")
    MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "3000"))
    MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))

VECTOR_DIM = 768


# ============================================================================
# Database Schema & Native SQLite Setup
# ============================================================================

def get_vec_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Connect to SQLite with WAL mode and sqlite-vec loaded."""
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_summary_tables(conn: sqlite3.Connection) -> None:
    """
    Layout the database schema cleanly with native SQL tables, indexes, and triggers.
    """
    cursor = conn.cursor()

    # 1. Main Document Summaries & Metadata Table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS document_summaries (
            doc_id INTEGER PRIMARY KEY,
            published_date TEXT,
            date_source TEXT,
            date_confidence REAL DEFAULT 0.0,
            summary TEXT NOT NULL,
            topics TEXT,
            model_name TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_doc_summaries_date ON document_summaries(published_date)"
    )

    # 2. Virtual Table for Vector Indexing on Summaries
    cursor.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_summaries USING vec0(
            doc_id integer primary key,
            embedding float[{VECTOR_DIM}] distance_metric=cosine
        )
        """
    )

    # 3. Native Trigger: Cascade deletion of vector index when a summary is deleted
    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_delete_summary_vector
        AFTER DELETE ON document_summaries
        BEGIN
            DELETE FROM vec_summaries WHERE doc_id = OLD.doc_id;
        END
        """
    )

    conn.commit()


# ============================================================================
# Step 1: Programmatic Metadata & Date Extractor
# ============================================================================

def extract_programmatic_date(html: str | None, markdown: str | None) -> dict[str, Any]:
    """
    Fast pre-pass in pure Python/regex (< 1 ms per doc).
    Extracts date from HTML microdata, OpenGraph, JSON-LD, or Markdown bylines.
    """
    html = html or ""
    markdown = markdown or ""

    # 1. HTML Microdata (e.g., <meta itemprop="datePublished" content="...">)
    m = re.search(r'<meta[^>]*itemprop=["\']datePublished["\'][^>]*content=["\']([^"\']+)["\']', html, re.I)
    if m:
        return {"date": m.group(1)[:10], "source": "meta_itemprop", "confidence": 0.99}

    # 2. OpenGraph article:published_time
    m = re.search(r'<meta[^>]*property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)["\']', html, re.I)
    if m:
        return {"date": m.group(1)[:10], "source": "opengraph", "confidence": 0.98}

    # 3. JSON-LD datePublished
    for j_match in re.finditer(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.DOTALL | re.I):
        try:
            data = json.loads(j_match.group(1))
            if isinstance(data, dict) and "datePublished" in data:
                return {"date": str(data["datePublished"])[:10], "source": "json_ld", "confidence": 0.98}
        except Exception:
            pass

    # 4. Known CMS timestamp tags (e.g. <p class="update-time">Cập nhật thứ tư, 03/01/2018</p>)
    m = re.search(r'<[^>]*class=["\'][^"\']*update-time[^"\']*["\'][^>]*>([^<]+)</', html, re.I)
    if m:
        date_text = m.group(1)
        num_m = re.search(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})', date_text)
        if num_m:
            d, month, y = num_m.groups()
            return {"date": f"{y}-{int(month):02d}-{int(d):02d}", "source": "html_class_updatetime", "confidence": 0.95}

    # 5. Markdown Vietnamese format bylines near header (e.g., "__ 21 Tháng 9 2026")
    lead_md = markdown[:2500]
    vn_m = re.search(r'(?:cập nhật|chi tiết|__)?\s*(\d{1,2})\s+Tháng\s+(\d{1,2})\s+(\d{4})', lead_md, re.I)
    if vn_m:
        d, month, y = vn_m.groups()
        return {"date": f"{y}-{int(month):02d}-{int(d):02d}", "source": "markdown_vn_byline", "confidence": 0.90}

    # 6. Numeric date pattern near publication keywords
    kw_m = re.search(r'(?:cập nhật|ngày đăng|thời gian|posted on|published)[\s:]*(?:thứ [^,]+,)?\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})', lead_md, re.I)
    if kw_m:
        d, month, y = kw_m.groups()
        return {"date": f"{y}-{int(month):02d}-{int(d):02d}", "source": "markdown_kw_date", "confidence": 0.85}

    return {"date": None, "source": None, "confidence": 0.0}


# ============================================================================
# Step 2: Unified LLM Client for Date & Summary Extraction
# ============================================================================

SYSTEM_PROMPT = """You are a precise document analyzer and summarizer.
Given a document excerpt and an optional metadata date hint, produce:
1. "published_date": The official publication or update date in ISO format (YYYY-MM-DD), or null if no publication date is found.
   - IMPORTANT: Do NOT extract match/event dates mentioned in the story (e.g. "Tournament held on Dec 5th"). Only extract when the article itself was posted.
   - Do NOT extract copyright years (e.g. "© 2015").
2. "date_confidence": A float from 0.0 to 1.0 indicating your confidence in the publication date.
3. "summary": A concise 2-3 sentence summary capturing the key subject, main details, and outcome. If the text is Vietnamese, write the summary in Vietnamese.
4. "topics": A list of 2 to 5 relevant keyword topics.

Respond ONLY with a valid JSON object matching this schema:
{
  "published_date": "YYYY-MM-DD" or null,
  "date_confidence": 0.95,
  "summary": "...",
  "topics": ["topic1", "topic2"]
}"""


class LLMClient:
    """Client for OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        endpoint_url: str = DEFAULT_LLM_URL,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: str = DEFAULT_LLM_API_KEY,
        timeout: float = 60.0,
    ):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.api_key = api_key
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def extract_and_summarize(
        self,
        text: str,
        title: str = "",
        url: str = "",
        date_hint: str | None = None,
    ) -> dict[str, Any]:
        """Send prompt to LLM and parse structured JSON response."""
        excerpt = text[:MAX_PROMPT_CHARS].strip()

        user_content = f"URL: {url}\nTitle: {title}\n"
        if date_hint:
            user_content += f"Metadata Hint (possible date): {date_hint}\n"
        user_content += f"\nDocument Excerpt:\n{excerpt}\n\nProvide the JSON analysis."

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 250,
            "response_format": {"type": "json_object"},
        }

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.post(self.endpoint_url, json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(f"Status {resp.status_code}", request=resp.request, response=resp)
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"]

                # Extract JSON block if surrounded by markdown code blocks
                clean_json = raw_text.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```", 1)[1].split("```", 1)[0].strip()

                parsed = json.loads(clean_json)
                return {
                    "published_date": parsed.get("published_date"),
                    "date_confidence": float(parsed.get("date_confidence", 0.0) or 0.0),
                    "summary": parsed.get("summary", ""),
                    "topics": parsed.get("topics", []),
                }
            except Exception as e:
                last_error = e
                backoff = min(15.0, 1.0 * (2**attempt))
                time.sleep(backoff)

        logger.warning(f"LLM failed after {MAX_RETRIES} attempts: {last_error}")
        return {
            "published_date": date_hint,
            "date_confidence": 0.5 if date_hint else 0.0,
            "summary": excerpt[:250].replace("\n", " ").strip() + "...",
            "topics": [],
        }

    def close(self) -> None:
        self._client.close()


# ============================================================================
# Step 3: Embedding Client for Summaries
# ============================================================================

class EmbeddingClient:
    """Client for OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        endpoint_url: str = DEFAULT_EMBED_URL,
        model_name: str = DEFAULT_EMBED_MODEL,
        api_key: str = DEFAULT_EMBED_API_KEY,
        timeout: float = 60.0,
    ):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model_name, "input": list(texts)}
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.post(self.endpoint_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                sorted_items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_items]
            except Exception as e:
                last_error = e
                time.sleep(min(15.0, 1.0 * (2**attempt)))
        raise RuntimeError(f"Embedding failed after {MAX_RETRIES} attempts: {last_error}")

    def embed_single(self, text: str) -> list[float]:
        res = self.embed_batch([text])
        return res[0]

    def close(self) -> None:
        self._client.close()


# ============================================================================
# Pipeline Runners
# ============================================================================

def run_summarization(
    db_path: str = DEFAULT_DB_PATH,
    llm_url: str = DEFAULT_LLM_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    api_key: str = DEFAULT_LLM_API_KEY,
    concurrency: int = 4,
    skip_existing: bool = True,
    limit: int | None = None,
) -> int:
    """Run programmatic pre-pass and LLM summarization pass."""
    conn = get_vec_connection(db_path)
    init_summary_tables(conn)
    cursor = conn.cursor()

    if skip_existing:
        query = """
            SELECT s.id, s.url, s.html, s.markdown, COALESCE(json_extract(s.metadata, '$.title'), '')
            FROM documents s
            WHERE length(s.markdown) > 0
              AND s.id NOT IN (SELECT doc_id FROM document_summaries)
            ORDER BY s.id
        """
    else:
        query = """
            SELECT s.id, s.url, s.html, s.markdown, COALESCE(json_extract(s.metadata, '$.title'), '')
            FROM documents s
            WHERE length(s.markdown) > 0
            ORDER BY s.id
        """

    rows = cursor.execute(query).fetchall()
    if limit is not None:
        rows = rows[:limit]

    total_docs = len(rows)
    print(f"\n--- Stage 1 & 2: Summarization & Date Extraction ---")
    print(f"Target documents: {total_docs} | LLM: {llm_model} at {llm_url}")

    if total_docs == 0:
        print("All documents already have summaries.")
        conn.close()
        return 0

    llm = LLMClient(endpoint_url=llm_url, model_name=llm_model, api_key=api_key)

    def process_doc(doc: tuple) -> tuple[int, dict[str, Any]]:
        doc_id, url, html, markdown, title = doc
        # Step 1: Instant Python regex/microdata pre-pass
        prog_meta = extract_programmatic_date(html, markdown)
        date_hint = prog_meta.get("date")

        # Step 2: Unified LLM pass (extracts date & generates summary)
        llm_result = llm.extract_and_summarize(
            text=markdown or "",
            title=title or "",
            url=url or "",
            date_hint=date_hint,
        )

        # Decide final date and source
        final_date = llm_result.get("published_date") or date_hint
        date_source = "llm_extracted"
        if prog_meta.get("confidence", 0.0) >= 0.95 and prog_meta.get("date") == final_date:
            date_source = prog_meta.get("source") or "html_meta"
        elif date_hint and not llm_result.get("published_date"):
            date_source = prog_meta.get("source") or "meta_hint"

        return doc_id, {
            "published_date": final_date,
            "date_source": date_source,
            "date_confidence": llm_result.get("date_confidence", 0.0),
            "summary": llm_result.get("summary", ""),
            "topics": json.dumps(llm_result.get("topics", []), ensure_ascii=False),
            "model_name": llm_model,
        }

    pbar = tqdm(total=total_docs, desc="Summarizing", unit="doc", dynamic_ncols=True)
    done_count = 0

    try:
        if concurrency <= 1:
            for row in rows:
                doc_id, res = process_doc(row)
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO document_summaries
                    (doc_id, published_date, date_source, date_confidence, summary, topics, model_name, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (doc_id, res["published_date"], res["date_source"], res["date_confidence"],
                     res["summary"], res["topics"], res["model_name"]),
                )
                conn.commit()
                done_count += 1
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                pending = deque()
                for row in rows:
                    pending.append(pool.submit(process_doc, row))
                    while len(pending) >= concurrency:
                        future = pending.popleft()
                        doc_id, res = future.result()
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO document_summaries
                            (doc_id, published_date, date_source, date_confidence, summary, topics, model_name, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                            """,
                            (doc_id, res["published_date"], res["date_source"], res["date_confidence"],
                             res["summary"], res["topics"], res["model_name"]),
                        )
                        conn.commit()
                        done_count += 1
                        pbar.update(1)

                while pending:
                    future = pending.popleft()
                    doc_id, res = future.result()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO document_summaries
                        (doc_id, published_date, date_source, date_confidence, summary, topics, model_name, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (doc_id, res["published_date"], res["date_source"], res["date_confidence"],
                         res["summary"], res["topics"], res["model_name"]),
                    )
                    conn.commit()
                    done_count += 1
                    pbar.update(1)
    finally:
        pbar.close()
        llm.close()
        conn.close()

    print(f"Successfully saved {done_count} document summaries to database.")
    return done_count


def run_summary_embeddings(
    db_path: str = DEFAULT_DB_PATH,
    embed_url: str = DEFAULT_EMBED_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    api_key: str = DEFAULT_EMBED_API_KEY,
    batch_size: int = 32,
    skip_existing: bool = True,
    limit: int | None = None,
) -> int:
    """Step 3: Embed document summaries into vec_summaries with sqlite-vec."""
    conn = get_vec_connection(db_path)
    init_summary_tables(conn)
    cursor = conn.cursor()

    if skip_existing:
        query = """
            SELECT s.doc_id, s.summary
            FROM document_summaries s
            WHERE s.doc_id NOT IN (SELECT doc_id FROM vec_summaries)
              AND length(s.summary) > 0
            ORDER BY s.doc_id
        """
    else:
        query = """
            SELECT s.doc_id, s.summary
            FROM document_summaries s
            WHERE length(s.summary) > 0
            ORDER BY s.doc_id
        """

    rows = cursor.execute(query).fetchall()
    if limit is not None:
        rows = rows[:limit]

    total_summaries = len(rows)
    print(f"\n--- Stage 3: Embedding Summaries into vec_summaries ---")
    print(f"Summaries to embed: {total_summaries} | Model: {embed_model} at {embed_url}")

    if total_summaries == 0:
        print("All summaries are already embedded in vec_summaries.")
        conn.close()
        return 0

    client = EmbeddingClient(endpoint_url=embed_url, model_name=embed_model, api_key=api_key)
    pbar = tqdm(total=total_summaries, desc="Embedding Summaries", unit="summary", dynamic_ncols=True)
    embedded_count = 0

    try:
        for i in range(0, total_summaries, batch_size):
            batch = rows[i : i + batch_size]
            doc_ids = [r[0] for r in batch]
            # Use task prefix for EmbeddingGemma
            texts = [DOC_PREFIX + r[1] for r in batch]
            embeddings = client.embed_batch(texts)

            insert_data = [
                (doc_id, sqlite_vec.serialize_float32(emb))
                for doc_id, emb in zip(doc_ids, embeddings)
            ]
            placeholders = ",".join("?" * len(doc_ids))
            conn.execute(f"DELETE FROM vec_summaries WHERE doc_id IN ({placeholders})", doc_ids)
            conn.executemany("INSERT INTO vec_summaries(doc_id, embedding) VALUES (?, ?)", insert_data)
            conn.commit()

            embedded_count += len(batch)
            pbar.update(len(batch))
    finally:
        pbar.close()
        client.close()
        conn.close()

    print(f"Successfully embedded {embedded_count} summaries into vec_summaries.")
    return embedded_count


def search_summaries(
    query_text: str,
    top_k: int = 5,
    db_path: str = DEFAULT_DB_PATH,
    embed_url: str = DEFAULT_EMBED_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    api_key: str = DEFAULT_EMBED_API_KEY,
) -> list[dict[str, Any]]:
    """Query vec_summaries and join with document_summaries using native SQL."""
    client = EmbeddingClient(endpoint_url=embed_url, model_name=embed_model, api_key=api_key)
    try:
        q_emb = client.embed_single(QUERY_PREFIX + query_text)
    finally:
        client.close()

    conn = get_vec_connection(db_path)
    serialized_q = sqlite_vec.serialize_float32(q_emb)

    sql = """
    SELECT
        s.doc_id,
        c.url,
        s.published_date,
        s.date_source,
        s.summary,
        s.topics,
        (1.0 - v.distance) AS score,
        v.distance
    FROM vec_summaries v
    JOIN document_summaries s ON s.doc_id = v.doc_id
    JOIN documents c ON c.id = s.doc_id
    WHERE v.embedding MATCH ? AND k = ?
    ORDER BY v.distance ASC
    """
    rows = conn.execute(sql, (serialized_q, top_k)).fetchall()
    conn.close()

    hits = []
    for doc_id, url, p_date, d_source, summary, topics, score, distance in rows:
        hits.append({
            "doc_id": doc_id,
            "url": url,
            "published_date": p_date,
            "date_source": d_source,
            "summary": summary,
            "topics": json.loads(topics) if topics else [],
            "score": score,
            "distance": distance,
        })
    return hits


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Document Summarization, Date Extraction, and Summary Vector Indexing."
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--stage", choices=("all", "summarize", "embed", "search"), default="all",
                        help="Pipeline stage to run (default: all)")

    # LLM Settings
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL, help=f"LLM Chat endpoint (default: {DEFAULT_LLM_URL})")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help=f"LLM model ID (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("--llm-api-key", default=DEFAULT_LLM_API_KEY, help="API key for LLM endpoint")
    parser.add_argument("--concurrency", type=int, default=2, help="Parallel in-flight requests to LLM (default: 2)")

    # Embedding Settings
    parser.add_argument("--embed-url", default=DEFAULT_EMBED_URL, help=f"Embedding endpoint (default: {DEFAULT_EMBED_URL})")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help=f"Embedding model ID (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--embed-api-key", default=DEFAULT_EMBED_API_KEY, help="API key for embedding endpoint")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding summaries (default: 32)")

    # Control
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N documents (for testing)")
    parser.add_argument("--reindex", action="store_true", help="Reindex existing records instead of skipping")
    parser.add_argument("--search", type=str, default="", help="Query to test against vec_summaries")
    parser.add_argument("--top-k", type=int, default=5, help="Top K search results (default: 5)")

    args = parser.parse_args()

    if args.search or args.stage == "search":
        query = args.search or "thông tin tuyển sinh"
        print(f"\nSearching summaries for: {query!r} (top {args.top_k} results)...")
        results = search_summaries(
            query_text=query,
            top_k=args.top_k,
            db_path=args.db,
            embed_url=args.embed_url,
            embed_model=args.embed_model,
            api_key=args.embed_api_key,
        )
        for i, hit in enumerate(results, 1):
            print(f"\n[{i}] Doc ID: {hit['doc_id']} | Date: {hit['published_date']} ({hit['date_source']}) | Score: {hit['score']:.4f}")
            print(f"    URL: {hit['url']}")
            print(f"    Summary: {hit['summary']}")
            if hit["topics"]:
                print(f"    Topics: {', '.join(hit['topics'])}")
        return

    if args.stage in ("all", "summarize"):
        run_summarization(
            db_path=args.db,
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            api_key=args.llm_api_key,
            concurrency=args.concurrency,
            skip_existing=not args.reindex,
            limit=args.limit,
        )

    if args.stage in ("all", "embed"):
        run_summary_embeddings(
            db_path=args.db,
            embed_url=args.embed_url,
            embed_model=args.embed_model,
            api_key=args.embed_api_key,
            batch_size=args.batch_size,
            skip_existing=not args.reindex,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
