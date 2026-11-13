"""
Build and maintain USearch HNSW vector index for Babel knowledge base.

Reads all 768-dim child vectors from `vec_chunks` in `stash.db`,
constructs an HNSW graph index with half-precision (F16) cosine metric,
and persists it to `stash.usearch`.

Supports:
- Full build from scratch
- Incremental update of newly embedded vectors
- Memory-mapped index verification
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import sqlite_vec
from usearch.index import Index, MetricKind, ScalarKind

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("usearch_builder")

DEFAULT_DB_PATH = str(PROJECT_ROOT / "stash.db")
DEFAULT_INDEX_PATH = str(PROJECT_ROOT / "stash.usearch")
VECTOR_DIM = 768


def create_usearch_index(dtype: ScalarKind = ScalarKind.F16) -> Index:
    """Instantiate a USearch index configured for 768-d cosine distance."""
    return Index(
        ndim=VECTOR_DIM,
        metric=MetricKind.Cos,
        dtype=dtype,
    )


def build_usearch_index(
    db_path: str = DEFAULT_DB_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    batch_size: int = 50000,
    incremental: bool = False,
    dtype: ScalarKind = ScalarKind.F16,
) -> Index:
    """
    Build or incrementally update a USearch HNSW index from `vec_chunks`.
    """
    t0 = time.time()
    logger.info("Opening database %s for vector extraction...", db_path)
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM vec_chunks")
    total_vectors = cur.fetchone()[0]
    logger.info("Total vectors in vec_chunks: %d", total_vectors)

    if total_vectors == 0:
        logger.warning("No vectors found in vec_chunks. Aborting index build.")
        conn.close()
        return create_usearch_index(dtype=dtype)

    index_file = Path(index_path)
    if incremental and index_file.exists() and index_file.stat().st_size > 0:
        logger.info("Loading existing USearch index from %s for incremental update...", index_path)
        index = create_usearch_index(dtype=dtype)
        index.load(str(index_file))
        logger.info("Loaded index with %d existing keys.", len(index))
        existing_keys = set(index.keys)
    else:
        logger.info("Creating fresh USearch HNSW index (ndim=%d, metric=cos, dtype=%s)...", VECTOR_DIM, dtype.name)
        index = create_usearch_index(dtype=dtype)
        existing_keys = set()

    # Stream vectors in chunks to keep memory usage minimal (no ORDER BY to avoid 1.3GB sort)
    cur.execute("SELECT chunk_id, embedding FROM vec_chunks")

    added_count = 0
    batch_keys = []
    batch_vecs = []

    fetch_start = time.time()
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break

        filtered_rows = [r for r in rows if r[0] not in existing_keys] if existing_keys else rows
        if not filtered_rows:
            continue

        keys = np.array([r[0] for r in filtered_rows], dtype=np.int64)
        raw_bytes = b"".join(r[1] for r in filtered_rows)
        vectors = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, VECTOR_DIM)

        index.add(keys, vectors)
        added_count += len(keys)
        logger.info("Indexed %d / %d vectors (%.1f%%, %d new added)...",
                    len(index), total_vectors, (len(index) / total_vectors) * 100, added_count)

    conn.close()

    if added_count > 0 or not index_file.exists():
        logger.info("Saving USearch index to disk at %s...", index_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        index.save(str(index_file))
        file_size_mb = index_file.stat().st_size / (1024 * 1024)
        logger.info("Successfully persisted index! Size: %.2f MB, Total keys: %d, Elapsed: %.2f seconds.",
                    file_size_mb, len(index), time.time() - t0)
    else:
        logger.info("Index already up to date (%d keys). No new vectors added.", len(index))

    return index


def verify_usearch_index(
    db_path: str = DEFAULT_DB_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    test_k: int = 30,
) -> None:
    """Verify that the memory-mapped USearch index works and benchmark query latency."""
    logger.info("Verifying memory-mapped USearch index from %s...", index_path)
    index = create_usearch_index()
    index.view(index_path)

    logger.info("Index view opened with %d keys.", len(index))

    # Fetch a sample vector from SQLite
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, embedding FROM vec_chunks LIMIT 1")
    row = cur.fetchone()
    conn.close()

    if not row:
        logger.warning("No vectors to test.")
        return

    sample_id, sample_blob = row
    q_vec = np.frombuffer(sample_blob, dtype=np.float32)

    # Warmup
    index.search(q_vec, test_k)

    # Benchmark 100 queries
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        matches = index.search(q_vec, test_k)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    p95_ms = np.percentile(latencies, 95)

    logger.info("=== USearch Benchmark Results (%d queries, k=%d) ===", len(latencies), test_k)
    logger.info("  Min Latency : %.3f ms", min_ms)
    logger.info("  Avg Latency : %.3f ms", avg_ms)
    logger.info("  P95 Latency : %.3f ms", p95_ms)
    logger.info("  Top matches : %s", list(matches.keys[:5]))


def main():
    parser = argparse.ArgumentParser(description="Build and manage USearch vector index for Babel.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to stash.db")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH, help="Path to stash.usearch")
    parser.add_argument("--incremental", action="store_true", help="Incrementally update existing index")
    parser.add_argument("--verify", action="store_true", help="Verify and benchmark index search")
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for vector ingestion")

    args = parser.parse_args()

    if args.verify:
        verify_usearch_index(args.db_path, args.index_path)
    else:
        build_usearch_index(
            db_path=args.db_path,
            index_path=args.index_path,
            batch_size=args.batch_size,
            incremental=args.incremental,
        )
        verify_usearch_index(args.db_path, args.index_path)


if __name__ == "__main__":
    main()
