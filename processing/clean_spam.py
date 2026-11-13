"""
clean_spam.py - Database Hygiene & Spam Scrubbing for stash.db.

Detects and purges SEO spam, gambling injections, non-FTU domain infiltrations,
and orphaned records across all tables in stash.db:
- documents
- fts_documents
- document_chunks
- vec_chunks
- document_summaries
- vec_documents
- vec_summaries
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.config import DEFAULT_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Known spam keywords and gambling domains injected via SEO hacks
SPAM_DOMAINS = [
    "casimbanz.com",
    "casino777esp.com",
    "casinoazurfrance.com",
    "casinodaysnz.com",
    "casinoextraavis.com",
    "casimba.com",
    "kubet",
    "thabet",
    "789bet",
    "hi88",
    "jun88",
    "new88",
    "shbet",
]

SPAM_KEYWORDS = [
    r"\bcasimba\b",
    r"\bcasino777\b",
    r"\bazurcasino\b",
    r"\bextra casino\b",
    r"\bcasino days\b",
    r"\bkèo nhà cái\b",
    r"\bđá gà thomo\b",
    r"\bđá gà trực tiếp\b",
    r"\bsoi cầu\b",
    r"\bgame bài đổi thưởng\b",
    r"\bnổ hũ\b",
    r"\bcá cược bóng đá\b",
    r"\bbaccarat online\b",
]

COMPILED_SPAM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SPAM_KEYWORDS]


def is_valid_domain(url: str) -> bool:
    """Validate whether the URL belongs to an official FTU university domain."""
    if not url:
        return False
    try:
        netloc = urlparse(url).netloc.lower()
        # Remove port if present
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        return netloc.endswith("ftu.edu.vn") or netloc == "ftu.edu.vn"
    except Exception:
        return False


def detect_spam(doc_id: int, url: str, title: str, markdown: str) -> list[str]:
    """Check a document for SEO spam, gambling links, or non-FTU domain."""
    reasons: list[str] = []

    # 1. Domain authority check
    if not is_valid_domain(url):
        reasons.append(f"Non-FTU domain: {url}")

    combined_text = f"{title or ''} {markdown or ''}".lower()

    # 2. Check known spam gambling external domains
    for domain in SPAM_DOMAINS:
        if domain in url.lower() or domain in combined_text:
            reasons.append(f"Spam domain match: {domain}")

    # 3. Regex keyword patterns
    for pat in COMPILED_SPAM_PATTERNS:
        match = pat.search(combined_text)
        if match:
            reasons.append(f"Spam pattern match: {match.group(0)}")

    return reasons


def find_spam_documents(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    """Scan documents and fts_documents for spam efficiently."""
    cursor = conn.cursor()
    spam_docs: dict[int, dict[str, Any]] = {}

    # 1. Non-FTU domains
    non_ftu = cursor.execute(
        "SELECT id, url, markdown FROM documents WHERE url NOT LIKE '%ftu.edu.vn%'"
    ).fetchall()
    for doc_id, url, md in non_ftu:
        spam_docs[doc_id] = {
            "doc_id": doc_id,
            "url": url,
            "title": "",
            "reasons": [f"Non-FTU domain: {url}"],
            "snippet": (md or "")[:200].replace("\n", " "),
        }

    # 2. Search for known spam keywords and domains via SQL LIKE
    keywords_to_check = [
        "casimba", "casino777", "azurcasino", "extra casino", "casino days",
        "baccarat", "kèo nhà cái", "đá gà", "soi cầu", "game bài", "nổ hũ"
    ]
    for keyword in keywords_to_check:
        matches = cursor.execute(
            """
            SELECT s.id, s.url, COALESCE(f.title, ''), s.markdown
            FROM documents s
            LEFT JOIN fts_documents f ON f.doc_id = s.id
            WHERE s.url LIKE ? OR s.markdown LIKE ?
            """,
            (f"%{keyword}%", f"%{keyword}%"),
        ).fetchall()
        for doc_id, url, title, md in matches:
            reasons = detect_spam(doc_id, url, title, md or "")
            if reasons:
                if doc_id not in spam_docs:
                    spam_docs[doc_id] = {
                        "doc_id": doc_id,
                        "url": url,
                        "title": title,
                        "reasons": reasons,
                        "snippet": (md or "")[:200].replace("\n", " "),
                    }
                else:
                    spam_docs[doc_id]["reasons"] = list(set(spam_docs[doc_id]["reasons"] + reasons))

    return spam_docs


def purge_documents(conn: sqlite3.Connection, doc_ids: list[int]) -> dict[str, int]:
    """Permanently delete specified documents from all relational, FTS, and vector tables."""
    if not doc_ids:
        return {}

    counts: dict[str, int] = {}
    placeholders = ",".join("?" for _ in doc_ids)

    # 1. Collect chunk_ids to purge from vec_chunks
    chunk_rows = conn.execute(
        f"SELECT chunk_id FROM document_chunks WHERE doc_id IN ({placeholders})", doc_ids
    ).fetchall()
    chunk_ids = [r[0] for r in chunk_rows]

    # 2. Purge from vec_chunks
    if chunk_ids:
        c_placeholders = ",".join("?" for _ in chunk_ids)
        try:
            cur = conn.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({c_placeholders})", chunk_ids)
            counts["vec_chunks"] = cur.rowcount
        except sqlite3.OperationalError as e:
            logger.warning("Could not delete from vec_chunks: %s", e)

    # 3. Purge from document_chunks
    cur = conn.execute(f"DELETE FROM document_chunks WHERE doc_id IN ({placeholders})", doc_ids)
    counts["document_chunks"] = cur.rowcount

    # 4. Purge from vec_documents
    try:
        cur = conn.execute(f"DELETE FROM vec_documents WHERE doc_id IN ({placeholders})", doc_ids)
        counts["vec_documents"] = cur.rowcount
    except sqlite3.OperationalError as e:
        logger.warning("Could not delete from vec_documents: %s", e)

    # 5. Purge from vec_summaries
    try:
        cur = conn.execute(f"DELETE FROM vec_summaries WHERE doc_id IN ({placeholders})", doc_ids)
        counts["vec_summaries"] = cur.rowcount
    except sqlite3.OperationalError as e:
        logger.warning("Could not delete from vec_summaries: %s", e)

    # 6. Purge from document_summaries
    cur = conn.execute(f"DELETE FROM document_summaries WHERE doc_id IN ({placeholders})", doc_ids)
    counts["document_summaries"] = cur.rowcount

    # 7. Purge from fts_documents
    cur = conn.execute(f"DELETE FROM fts_documents WHERE doc_id IN ({placeholders})", doc_ids)
    counts["fts_documents"] = cur.rowcount

    # 8. Purge from documents
    cur = conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", doc_ids)
    counts["documents"] = cur.rowcount

    conn.commit()
    return counts


def clean_spam_pipeline(db_path: str = DEFAULT_DB_PATH, dry_run: bool = False, vacuum: bool = True) -> None:
    """Run full detection and purging pipeline on stash.db."""
    logger.info("Opening database at %s...", db_path)
    conn = sqlite3.connect(db_path, timeout=60.0)

    if sqlite_vec is not None:
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as e:
            logger.warning("Could not load sqlite-vec: %s", e)

    spam_docs = find_spam_documents(conn)
    logger.info("Identified %d spam/unauthorized documents.", len(spam_docs))

    for doc_id, item in spam_docs.items():
        logger.info(
            "  Doc ID %d: %s | Title: %s\n    Reasons: %s\n    Snippet: %s",
            doc_id,
            item["url"],
            item["title"],
            ", ".join(item["reasons"]),
            item["snippet"],
        )

    if not spam_docs:
        logger.info("No spam documents found. Database is clean!")
        conn.close()
        return

    if dry_run:
        logger.info("DRY-RUN mode enabled. No changes made.")
        conn.close()
        return

    doc_ids_to_purge = list(spam_docs.keys())
    logger.info("Purging %d documents...", len(doc_ids_to_purge))
    counts = purge_documents(conn, doc_ids_to_purge)
    for table, count in counts.items():
        logger.info("  Deleted %d rows from %s", count, table)

    # Optimize FTS5
    logger.info("Optimizing FTS5 index (fts_documents)...")
    try:
        conn.execute("INSERT INTO fts_documents(fts_documents) VALUES('optimize')")
        conn.commit()
        logger.info("FTS5 index optimized.")
    except Exception as e:
        logger.warning("FTS optimize failed: %s", e)

    conn.close()

    if vacuum:
        logger.info("Running VACUUM to reclaim space and rebuild pages...")
        v_conn = sqlite3.connect(db_path, timeout=120.0)
        t0 = time.time()
        v_conn.execute("VACUUM")
        v_conn.close()
        logger.info("VACUUM completed in %.2fs.", time.time() - t0)

    logger.info("Spam scrubbing pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purge spam and corrupted data from stash.db")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to stash.db")
    parser.add_argument("--dry-run", action="store_true", help="Scan only without deleting")
    parser.add_argument("--no-vacuum", action="store_true", help="Skip database VACUUM")
    args = parser.parse_args()

    clean_spam_pipeline(db_path=args.db_path, dry_run=args.dry_run, vacuum=not args.no_vacuum)
