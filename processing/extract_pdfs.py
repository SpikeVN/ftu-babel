"""
Batch download and extract PDF documents using pymupdf4llm for stash.db.

Features:
- Downloads 1,020 PDFs into `downloads/pdfs/{doc_id}.pdf`.
- Supports Google Drive direct download and virus-scan bypass.
- Triages documents into DIGITAL vs SCANNED:
    * Digital PDFs: Extracted via `pymupdf4llm` to preserve full Markdown tables and structure.
    * Scanned PDFs: Marked as `scanned` in `document_stubs` with `local_path` preserved for OCR.
    * Unavailable / Failed: Marked with error status.
- Updates `documents.markdown` and triggers SQLite FTS5 index update automatically.
- Resumable: Skips already-downloaded valid PDFs unless --force is given.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pymupdf
import pymupdf4llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("extract_pdfs")

# Suppress MuPDF C-level warnings
try:
    pymupdf.TOOLS.mupdf_display_errors(False)
except Exception:
    pass

# Output directory for raw downloaded PDFs
PDF_DIR = Path("downloads/pdfs")

# Minimum characters per page to consider a document "digital" (has selectable text layer)
MIN_CHARS_PER_PAGE = 30
MIN_TOTAL_CHARS = 80


def get_drive_file_id(url: str) -> str | None:
    """Extract Google Drive file ID from various URL patterns."""
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def get_drive_confirm_token(html: str) -> str | None:
    """Extract confirm token for large Google Drive files."""
    m = re.search(r"confirm=([0-9A-Za-z_]+)", html)
    if m:
        return m.group(1)
    m = re.search(r'name=["\']confirm["\']\s+value=["\']([0-9A-Za-z_]+)["\']', html)
    if m:
        return m.group(1)
    return None


async def download_pdf(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    drive_semaphore: asyncio.Semaphore,
    doc_id: int,
    url: str,
    dest_path: Path,
    force: bool = False,
) -> tuple[bool, str]:
    """
    Downloads a PDF to dest_path with throttling and retry logic for Google Drive.
    Returns (success, error_or_info_msg).
    """
    if not force and dest_path.exists() and dest_path.stat().st_size > 1000:
        # Check if already a valid PDF
        try:
            with dest_path.open("rb") as f:
                header = f.read(5)
                if header.startswith(b"%PDF"):
                    return True, "cached"
        except Exception:
            pass

    is_gdrive = "drive.google.com" in url or "docs.google.com" in url
    active_sem = drive_semaphore if is_gdrive else semaphore

    async with active_sem:
        max_attempts = 3 if is_gdrive else 2
        for attempt in range(1, max_attempts + 1):
            try:
                if is_gdrive:
                    file_id = get_drive_file_id(url)
                    if not file_id:
                        return False, f"Could not parse Google Drive ID from {url}"
                    dl_url = f"https://drive.google.com/uc?id={file_id}&export=download"
                    # Gentle pacing between Google requests
                    await asyncio.sleep(0.25)
                else:
                    dl_url = url

                resp = await client.get(dl_url)

                # Check if redirected to Google ServiceLogin (rate-limit / bot check)
                if is_gdrive and "accounts.google.com" in str(resp.url):
                    if attempt < max_attempts:
                        await asyncio.sleep(1.5 * attempt)
                        continue
                    return False, "Google Drive auth wall (login required or rate limited)"

                # Check if Google Drive returned virus warning confirmation page
                if (
                    resp.status_code == 200
                    and not resp.content.startswith(b"%PDF")
                    and is_gdrive
                ):
                    html_text = resp.text
                    token = get_drive_confirm_token(html_text)
                    if token:
                        file_id = get_drive_file_id(url)
                        confirm_url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={token}"
                        resp = await client.get(confirm_url)

                if resp.status_code == 429:
                    if attempt < max_attempts:
                        await asyncio.sleep(2.0 * attempt)
                        continue
                    return False, "HTTP 429 Rate Limited"

                if resp.status_code != 200:
                    return False, f"HTTP {resp.status_code}"

                content = resp.content
                if not content.startswith(b"%PDF"):
                    if len(content) < 3000 and (b"<html" in content.lower() or b"<!doctype" in content.lower()):
                        if attempt < max_attempts and is_gdrive:
                            await asyncio.sleep(1.5 * attempt)
                            continue
                        return False, "Returned HTML instead of PDF (auth/permission wall or deleted file)"
                    return False, f"Invalid PDF header: {content[:10]!r}"

                # Save to disk
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with dest_path.open("wb") as f:
                    f.write(content)

                return True, f"downloaded ({len(content)} bytes)"

            except httpx.TimeoutException:
                if attempt < max_attempts:
                    await asyncio.sleep(1.0)
                    continue
                return False, "Download timeout"
            except Exception as e:
                if attempt < max_attempts:
                    await asyncio.sleep(1.0)
                    continue
                return False, f"Download error: {e}"

    return False, "Failed after max retries"


def sanitize_text(text: str) -> str:
    """Sanitize unpaired surrogates and null bytes for UTF-8 / SQLite storage."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    return text.encode("utf-8", errors="replace").decode("utf-8")


def parse_pdf_date(d: str) -> str | None:
    """Normalize PDF date string (D:YYYYMMDDHHmmSS) to ISO 8601."""
    if not d:
        return None
    d = d.strip()
    if d.startswith("D:"):
        d = d[2:]
    m = re.match(r"^(\d{4})(\d{2})(\d{2})(?:(\d{2})(\d{2})(\d{2}))?", d)
    if m:
        year, month, day, hour, minute, second = m.groups()
        if hour is not None:
            return f"{year}-{month}-{day}T{hour}:{minute}:{second}"
        return f"{year}-{month}-{day}"
    return d


def extract_content(pdf_path: Path) -> tuple[str, str, int, int, dict]:
    """
    Inspects and extracts text / tables and metadata from a local PDF using pymupdf4llm.
    Returns:
        (category, markdown_or_empty, page_count, raw_char_count, metadata_dict)
        where category is 'digital', 'scanned', or 'corrupt'
    """
    empty_meta = {"created_at": None, "modified_at": None, "author": None, "producer": None}
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        return "corrupt", "", 0, 0, empty_meta

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        return "corrupt", "", 0, 0, empty_meta

    # Extract PDF embedded metadata
    raw_meta = doc.metadata or {}
    meta_dict = {
        "created_at": parse_pdf_date(raw_meta.get("creationDate")),
        "modified_at": parse_pdf_date(raw_meta.get("modDate")),
        "author": raw_meta.get("author") or None,
        "producer": raw_meta.get("producer") or raw_meta.get("creator") or None,
    }

    # Quick check for selectable text layer
    total_raw_chars = sum(len(p.get_text().strip()) for p in doc)
    avg_chars = total_raw_chars / page_count

    if avg_chars < MIN_CHARS_PER_PAGE and total_raw_chars < MIN_TOTAL_CHARS:
        # Document is scanned (raster images)
        doc.close()
        return "scanned", "", page_count, total_raw_chars, meta_dict

    # Digital document with selectable text layer -> extract full Markdown
    try:
        md_text = pymupdf4llm.to_markdown(doc, page_chunks=False, write_images=False)
        doc.close()
        md_text = sanitize_text(md_text)
        return "digital", md_text, page_count, len(md_text), meta_dict
    except Exception as e:
        doc.close()
        logger.warning("pymupdf4llm error on %s: %s", pdf_path, e)
        # Fallback to plain text if pymupdf4llm fails on layout
        doc = pymupdf.open(str(pdf_path))
        plain = "\n\n".join(p.get_text() for p in doc)
        doc.close()
        plain = sanitize_text(plain)
        return "digital", plain, page_count, len(plain), meta_dict


async def process_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    drive_semaphore: asyncio.Semaphore,
    doc_id: int,
    url: str,
    title: str,
    force: bool,
) -> dict:
    dest_path = PDF_DIR / f"{doc_id}.pdf"

    # Step 1: Download
    ok, dl_msg = await download_pdf(client, semaphore, drive_semaphore, doc_id, url, dest_path, force=force)
    if not ok:
        return {
            "doc_id": doc_id,
            "status": "unavailable",
            "category": "failed",
            "message": dl_msg,
            "chars": 0,
            "pages": 0,
            "local_path": None,
            "markdown": None,
            "meta": {"created_at": None, "modified_at": None, "author": None, "producer": None},
        }

    # Step 2: Extract in thread pool (CPU-bound)
    category, markdown, pages, chars, meta = await asyncio.to_thread(extract_content, dest_path)

    if category == "digital":
        status = "extracted"
    elif category == "scanned":
        status = "scanned"
    else:
        status = "corrupt"

    return {
        "doc_id": doc_id,
        "status": status,
        "category": category,
        "message": dl_msg,
        "chars": chars,
        "pages": pages,
        "local_path": str(dest_path),
        "markdown": markdown,
        "meta": meta,
    }


async def run_pipeline(
    db_path: str = "stash.db",
    concurrency: int = 15,
    drive_concurrency: int = 4,
    limit: int | None = None,
    doc_id: int | None = None,
    force: bool = False,
) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if doc_id:
        cur.execute(
            "SELECT doc_id, url, title, status FROM document_stubs WHERE doc_id = ?",
            (doc_id,),
        )
    elif force:
        cur.execute(
            "SELECT doc_id, url, title, status FROM document_stubs WHERE type = 'application/pdf'"
        )
    else:
        # Process pending, corrupt, or unavailable stubs (skip already extracted or confirmed scanned)
        cur.execute(
            """
            SELECT doc_id, url, title, status
            FROM document_stubs
            WHERE type = 'application/pdf' AND status NOT IN ('extracted', 'scanned')
            """
        )

    rows = cur.fetchall()
    if limit:
        rows = rows[:limit]

    total = len(rows)
    logger.info("Found %d PDF stubs to process (concurrency=%d, drive_concurrency=%d)...", total, concurrency, drive_concurrency)
    if total == 0:
        logger.info("No pending PDFs to process.")
        conn.close()
        return

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    limits = httpx.Limits(max_keepalive_connections=concurrency * 2, max_connections=concurrency)
    timeout = httpx.Timeout(30.0, connect=10.0)
    semaphore = asyncio.Semaphore(concurrency)
    drive_semaphore = asyncio.Semaphore(drive_concurrency)

    stats = {"digital": 0, "scanned": 0, "unavailable": 0, "corrupt": 0, "total_chars": 0}
    completed = 0
    t0 = time.time()

    async with httpx.AsyncClient(
        verify=False,
        follow_redirects=True,
        headers=headers,
        limits=limits,
        timeout=timeout,
    ) as client:
        # Create tasks
        tasks = [
            process_one(client, semaphore, drive_semaphore, r_id, r_url, r_title, force)
            for r_id, r_url, r_title, _ in rows
        ]

        batch_updates = []
        for fut in asyncio.as_completed(tasks):
            res = await fut
            completed += 1
            cat = res["category"]
            stats[cat if cat in stats else "unavailable"] += 1
            stats["total_chars"] += res["chars"]

            batch_updates.append(res)

            # Flush batch to SQLite periodically
            if len(batch_updates) >= 20 or completed == total:
                for item in batch_updates:
                    d_id = item["doc_id"]
                    st = item["status"]
                    lp = item["local_path"]
                    ch = item["chars"]
                    md = item["markdown"]

                    meta = item.get("meta") or {}
                    c_at = meta.get("created_at")
                    m_at = meta.get("modified_at")
                    auth = meta.get("author")
                    prod = meta.get("producer")

                    try:
                        # Update document_stubs
                        cur.execute(
                            """
                            UPDATE document_stubs
                            SET status = ?, local_path = ?, extracted_chars = ?,
                                file_created_at = COALESCE(?, file_created_at),
                                file_modified_at = COALESCE(?, file_modified_at),
                                file_author = COALESCE(?, file_author),
                                file_producer = COALESCE(?, file_producer)
                            WHERE doc_id = ?
                            """,
                            (st, lp, ch, c_at, m_at, auth, prod, d_id),
                        )

                        # Update documents table
                        if st == "extracted" and md:
                            cur.execute(
                                """
                                UPDATE documents
                                SET markdown = ?, success = 1,
                                    metadata = json_set(
                                        COALESCE(metadata, '{}'),
                                        '$.created_at', COALESCE(?, json_extract(metadata, '$.created_at')),
                                        '$.modified_at', COALESCE(?, json_extract(metadata, '$.modified_at')),
                                        '$.author', COALESCE(?, json_extract(metadata, '$.author')),
                                        '$.producer', COALESCE(?, json_extract(metadata, '$.producer'))
                                    )
                                WHERE id = ?
                                """,
                                (md, c_at, m_at, auth, prod, d_id),
                            )
                        elif c_at or m_at or auth or prod:
                            cur.execute(
                                """
                                UPDATE documents
                                SET metadata = json_set(
                                        COALESCE(metadata, '{}'),
                                        '$.created_at', COALESCE(?, json_extract(metadata, '$.created_at')),
                                        '$.modified_at', COALESCE(?, json_extract(metadata, '$.modified_at')),
                                        '$.author', COALESCE(?, json_extract(metadata, '$.author')),
                                        '$.producer', COALESCE(?, json_extract(metadata, '$.producer'))
                                    )
                                WHERE id = ?
                                """,
                                (c_at, m_at, auth, prod, d_id),
                            )
                    except Exception as err:
                        logger.error("Error saving doc #%d to database: %s", d_id, err)
                        cur.execute(
                            "UPDATE document_stubs SET status = 'error' WHERE doc_id = ?",
                            (d_id,),
                        )

                conn.commit()
                batch_updates.clear()

            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / max(0.1, elapsed)
                logger.info(
                    "Progress: %d/%d (%.1f%%) | Digital: %d | Scanned: %d | Unavail: %d | Rate: %.1f docs/s",
                    completed,
                    total,
                    completed / total * 100,
                    stats["digital"],
                    stats["scanned"],
                    stats["unavailable"],
                    rate,
                )

    conn.close()
    elapsed = time.time() - t0
    logger.info("=== PDF Processing Completed in %.1fs ===", elapsed)
    logger.info("  Digital (Extracted with Markdown Tables) : %d", stats["digital"])
    logger.info("  Scanned (Saved for OCR Pipeline)         : %d", stats["scanned"])
    logger.info("  Unavailable / Auth-walled / 404          : %d", stats["unavailable"])
    logger.info("  Corrupt                                  : %d", stats["corrupt"])
    logger.info("  Total Extracted Characters               : %d", stats["total_chars"])


def main():
    parser = argparse.ArgumentParser(description="Batch download & extract PDF documents for stash.db")
    parser.add_argument("--db", default="stash.db", help="Path to database (default: stash.db)")
    parser.add_argument("--workers", type=int, default=15, help="Concurrent workers (default: 15)")
    parser.add_argument("--limit", type=int, help="Limit number of PDFs to process")
    parser.add_argument("--doc-id", type=int, help="Process a specific doc_id only")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-extract")

    args = parser.parse_args()
    asyncio.run(
        run_pipeline(
            db_path=args.db,
            concurrency=args.workers,
            limit=args.limit,
            doc_id=args.doc_id,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
