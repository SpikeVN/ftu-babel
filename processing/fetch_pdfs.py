"""
Dedicated batch fetcher/downloader for PDF documents in stash.db.

Focuses solely on robust, resumable downloading to `downloads/pdfs/{doc_id}.pdf`:
- Skips already-downloaded valid PDFs.
- Dedicated Google Drive throttling (concurrency=4 with inter-request delays).
- Dropbox direct download parameter conversion (`dl=1`).
- Handles Google Drive large file virus-scan confirmations.
- Automatically extracts and saves PDF creation/modification dates and author metadata.
- Updates `document_stubs` with `local_path` and metadata.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_pdfs")

# Suppress MuPDF C-level warnings
try:
    pymupdf.TOOLS.mupdf_display_errors(False)
except Exception:
    pass

PDF_DIR = Path("downloads/pdfs")


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


def get_drive_file_id(url: str) -> str | None:
    """Extract Google Drive file ID from URL."""
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


async def fetch_one(
    client: httpx.AsyncClient,
    general_sem: asyncio.Semaphore,
    drive_sem: asyncio.Semaphore,
    doc_id: int,
    url: str,
    title: str,
    force: bool,
) -> dict:
    dest_path = PDF_DIR / f"{doc_id}.pdf"

    # Check if already cached
    if not force and dest_path.exists() and dest_path.stat().st_size > 1000:
        try:
            with dest_path.open("rb") as f:
                if f.read(5).startswith(b"%PDF"):
                    # Extract metadata from cached file
                    doc = pymupdf.open(str(dest_path))
                    rmeta = doc.metadata or {}
                    meta = {
                        "created_at": parse_pdf_date(rmeta.get("creationDate")),
                        "modified_at": parse_pdf_date(rmeta.get("modDate")),
                        "author": rmeta.get("author") or None,
                        "producer": rmeta.get("producer") or rmeta.get("creator") or None,
                    }
                    doc.close()
                    return {
                        "doc_id": doc_id,
                        "success": True,
                        "status": "cached",
                        "size": dest_path.stat().st_size,
                        "local_path": str(dest_path),
                        "meta": meta,
                    }
        except Exception:
            pass

    is_gdrive = "drive.google.com" in url or "docs.google.com" in url
    is_dropbox = "dropbox.com" in url
    active_sem = drive_sem if is_gdrive else general_sem

    async with active_sem:
        max_attempts = 3 if is_gdrive else 2
        for attempt in range(1, max_attempts + 1):
            try:
                if is_gdrive:
                    file_id = get_drive_file_id(url)
                    if not file_id:
                        return {"doc_id": doc_id, "success": False, "status": "bad_url", "error": f"Bad Drive URL {url}"}
                    dl_url = f"https://drive.google.com/uc?id={file_id}&export=download"
                    await asyncio.sleep(0.3)
                elif is_dropbox:
                    dl_url = re.sub(r"[?&]dl=0", "", url)
                    sep = "&" if "?" in dl_url else "?"
                    dl_url = f"{dl_url}{sep}dl=1"
                else:
                    dl_url = url

                resp = await client.get(dl_url)

                # Fail fast on Google Drive "download disabled by owner"
                if is_gdrive and (b"Can't download file" in resp.content or b"permission to download this file" in resp.content):
                    return {"doc_id": doc_id, "success": False, "status": "download_disabled_by_owner", "error": "Owner disabled download"}

                # Fail fast on 404 Not Found
                if resp.status_code == 404:
                    return {"doc_id": doc_id, "success": False, "status": "not_found_404", "error": "HTTP 404"}

                # Fail fast on 403 Forbidden
                if resp.status_code == 403:
                    return {"doc_id": doc_id, "success": False, "status": "forbidden_403", "error": "HTTP 403"}

                # Check if redirected to Google login
                if is_gdrive and "accounts.google.com" in str(resp.url):
                    if attempt < max_attempts:
                        await asyncio.sleep(2.0 * attempt)
                        continue
                    return {"doc_id": doc_id, "success": False, "status": "auth_wall", "error": "Google Drive auth wall"}

                # Check Google virus confirmation
                if resp.status_code == 200 and not resp.content.startswith(b"%PDF") and is_gdrive:
                    html_text = resp.text
                    token = get_drive_confirm_token(html_text)
                    if token:
                        file_id = get_drive_file_id(url)
                        confirm_url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={token}"
                        resp = await client.get(confirm_url)

                if resp.status_code == 429:
                    if attempt < max_attempts:
                        await asyncio.sleep(3.0 * attempt)
                        continue
                    return {"doc_id": doc_id, "success": False, "status": "rate_limited", "error": "HTTP 429"}

                if resp.status_code != 200:
                    return {"doc_id": doc_id, "success": False, "status": f"http_{resp.status_code}", "error": f"HTTP {resp.status_code}"}

                content = resp.content
                if not content.startswith(b"%PDF"):
                    if len(content) < 3000 and (b"<html" in content.lower() or b"<!doctype" in content.lower()):
                        if attempt < max_attempts and is_gdrive:
                            await asyncio.sleep(2.0 * attempt)
                            continue
                        return {"doc_id": doc_id, "success": False, "status": "html_returned", "error": "HTML returned instead of PDF"}
                    return {"doc_id": doc_id, "success": False, "status": "invalid_header", "error": f"Invalid header: {content[:10]!r}"}

                # Save file
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with dest_path.open("wb") as f:
                    f.write(content)

                # Extract metadata
                meta = {"created_at": None, "modified_at": None, "author": None, "producer": None}
                try:
                    doc = pymupdf.open(str(dest_path))
                    rmeta = doc.metadata or {}
                    meta = {
                        "created_at": parse_pdf_date(rmeta.get("creationDate")),
                        "modified_at": parse_pdf_date(rmeta.get("modDate")),
                        "author": rmeta.get("author") or None,
                        "producer": rmeta.get("producer") or rmeta.get("creator") or None,
                    }
                    doc.close()
                except Exception:
                    pass

                return {
                    "doc_id": doc_id,
                    "success": True,
                    "status": "downloaded",
                    "size": len(content),
                    "local_path": str(dest_path),
                    "meta": meta,
                }

            except httpx.TimeoutException:
                if attempt < max_attempts:
                    await asyncio.sleep(1.5)
                    continue
                return {"doc_id": doc_id, "success": False, "status": "timeout", "error": "Timeout"}
            except Exception as e:
                if attempt < max_attempts:
                    await asyncio.sleep(1.5)
                    continue
                return {"doc_id": doc_id, "success": False, "status": "error", "error": str(e)}

    return {"doc_id": doc_id, "success": False, "status": "failed", "error": "Max retries exceeded"}


async def run_fetch(
    db_path: str = "stash.db",
    force: bool = False,
    limit: int | None = None,
) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT doc_id, url, title, status, local_path
        FROM document_stubs
        WHERE type = 'application/pdf'
        ORDER BY doc_id ASC
        """
    )
    all_rows = cur.fetchall()

    targets = []
    cached_count = 0
    for r_id, r_url, r_title, r_status, r_lp in all_rows:
        dest_p = PDF_DIR / f"{r_id}.pdf"
        if not force and dest_p.exists() and dest_p.stat().st_size > 1000:
            cached_count += 1
        else:
            targets.append((r_id, r_url, r_title))

    if limit:
        targets = targets[:limit]

    total_needed = len(targets)
    logger.info(
        "Total PDF Stubs: %d | Already Cached on Disk: %d | Remaining to Fetch: %d",
        len(all_rows),
        cached_count,
        total_needed,
    )

    if total_needed == 0:
        logger.info("All PDFs are already fetched and cached on disk.")
        conn.close()
        return

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=12)
    timeout = httpx.Timeout(30.0, connect=10.0)

    general_sem = asyncio.Semaphore(6)
    drive_sem = asyncio.Semaphore(4)

    stats = {"downloaded": 0, "failed": 0, "bytes": 0}
    completed = 0
    t0 = time.time()

    async with httpx.AsyncClient(
        verify=False,
        follow_redirects=True,
        headers=headers,
        limits=limits,
        timeout=timeout,
    ) as client:
        tasks = [
            fetch_one(client, general_sem, drive_sem, d_id, url, title, force)
            for d_id, url, title in targets
        ]

        batch_updates = []
        for fut in asyncio.as_completed(tasks):
            res = await fut
            completed += 1

            if res["success"]:
                stats["downloaded"] += 1
                stats["bytes"] += res.get("size", 0)
            else:
                stats["failed"] += 1

            batch_updates.append(res)

            # Flush batch to SQLite periodically
            if len(batch_updates) >= 15 or completed == total_needed:
                for item in batch_updates:
                    d_id = item["doc_id"]
                    if item["success"]:
                        lp = item["local_path"]
                        meta = item.get("meta") or {}
                        c_at = meta.get("created_at")
                        m_at = meta.get("modified_at")
                        auth = meta.get("author")
                        prod = meta.get("producer")

                        cur.execute(
                            """
                            UPDATE document_stubs
                            SET local_path = ?,
                                file_created_at = COALESCE(?, file_created_at),
                                file_modified_at = COALESCE(?, file_modified_at),
                                file_author = COALESCE(?, file_author),
                                file_producer = COALESCE(?, file_producer)
                            WHERE doc_id = ?
                            """,
                            (lp, c_at, m_at, auth, prod, d_id),
                        )
                        # If stub was previously unavailable, reset to pending for extraction
                        cur.execute(
                            """
                            UPDATE document_stubs
                            SET status = 'pending'
                            WHERE doc_id = ? AND status = 'unavailable'
                            """,
                            (d_id,),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE document_stubs
                            SET status = 'unavailable'
                            WHERE doc_id = ? AND status != 'extracted'
                            """,
                            (d_id,),
                        )

                conn.commit()
                batch_updates.clear()

            if completed % 20 == 0 or completed == total_needed:
                elapsed = time.time() - t0
                rate = completed / max(0.1, elapsed)
                mb = stats["bytes"] / (1024 * 1024)
                logger.info(
                    "Fetch Progress: %d/%d (%.1f%%) | Downloaded: %d (%.1f MB) | Failed/Unavail: %d | Rate: %.1f files/s",
                    completed,
                    total_needed,
                    completed / total_needed * 100,
                    stats["downloaded"],
                    mb,
                    stats["failed"],
                    rate,
                )

    conn.close()
    elapsed = time.time() - t0
    logger.info("=== PDF Fetching Completed in %.1fs ===", elapsed)
    logger.info("  Newly Downloaded : %d (%.1f MB)", stats["downloaded"], stats["bytes"] / (1024 * 1024))
    logger.info("  Failed / Unavailable : %d", stats["failed"])
    logger.info("  Total on Disk Now    : %d / %d", cached_count + stats["downloaded"], len(all_rows))


def main():
    parser = argparse.ArgumentParser(description="Fetch and download all PDF documents to disk")
    parser.add_argument("--db", default="stash.db", help="Path to database")
    parser.add_argument("--force", action="store_true", help="Force re-download cached PDFs")
    parser.add_argument("--limit", type=int, help="Limit number of PDFs to fetch")
    args = parser.parse_args()

    asyncio.run(run_fetch(db_path=args.db, force=args.force, limit=args.limit))


if __name__ == "__main__":
    main()
