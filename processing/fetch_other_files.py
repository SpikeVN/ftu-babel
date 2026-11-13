"""
Batch downloader for non-PDF document stubs in stash.db.

Handles:
- Google Docs: Exported directly as DOCX (`export?format=docx`) -> `downloads/docx/{doc_id}.docx`
- Google Sheets: Exported as XLSX (`export?format=xlsx`) -> `downloads/excel/{doc_id}.xlsx`
- Word Documents (.docx, .doc) -> `downloads/docx/` and `downloads/doc/`
- Excel Spreadsheets (.xlsx, .xls) -> `downloads/excel/`
- Google Forms (HTML survey snapshot) -> `downloads/forms/{doc_id}.html`
- Google Drive Folders (directory snapshot) -> `downloads/folders/{doc_id}.html`
- Archives (.zip, .rar) -> `downloads/archives/`
- Images & Media (.jpg, .png, .mp4) -> `downloads/images/` and `downloads/media/`
- Text (.txt) -> `downloads/text/`

Updates `document_stubs.local_path` and `status` in stash.db.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_other")

DOWNLOADS_DIR = Path("downloads")


def get_drive_id(url: str) -> str | None:
    """Extract ID from Google Drive or Google Docs URL."""
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def get_target_path_and_url(doc_id: int, mime_type: str, raw_url: str) -> tuple[Path, str, str]:
    """
    Determines destination Path, effective download URL, and file category.
    Returns (dest_path, dl_url, category).
    """
    fid = get_drive_id(raw_url)

    # 1. Google Docs -> export as DOCX
    if mime_type == "application/vnd.google-apps.document":
        dest = DOWNLOADS_DIR / "docx" / f"{doc_id}.docx"
        if fid:
            dl_url = f"https://docs.google.com/document/d/{fid}/export?format=docx"
        else:
            dl_url = raw_url
        return dest, dl_url, "gdoc_docx"

    # 2. Google Sheets -> export as XLSX or HTML if pubhtml
    if mime_type == "application/vnd.google-apps.spreadsheet":
        if "/pubhtml" in raw_url:
            dest = DOWNLOADS_DIR / "excel" / f"{doc_id}.html"
            return dest, raw_url, "gsheet_html"
        dest = DOWNLOADS_DIR / "excel" / f"{doc_id}.xlsx"
        if fid:
            dl_url = f"https://docs.google.com/spreadsheets/d/{fid}/export?format=xlsx"
        else:
            dl_url = raw_url
        return dest, dl_url, "gsheet_xlsx"

    # 3. Google Forms -> save HTML
    if mime_type == "application/vnd.google-apps.form":
        dest = DOWNLOADS_DIR / "forms" / f"{doc_id}.html"
        return dest, raw_url, "gform_html"

    # 4. Google Drive Folders -> save HTML
    if mime_type == "application/vnd.google-apps.folder":
        dest = DOWNLOADS_DIR / "folders" / f"{doc_id}.html"
        return dest, raw_url, "gfolder_html"

    # 5. Word Documents
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        dest = DOWNLOADS_DIR / "docx" / f"{doc_id}.docx"
        return dest, raw_url, "docx"
    if mime_type == "application/msword":
        dest = DOWNLOADS_DIR / "doc" / f"{doc_id}.doc"
        return dest, raw_url, "doc"

    # 6. Excel Spreadsheets
    if mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        dest = DOWNLOADS_DIR / "excel" / f"{doc_id}.xlsx"
        return dest, raw_url, "xlsx"
    if mime_type == "application/vnd.ms-excel":
        dest = DOWNLOADS_DIR / "excel" / f"{doc_id}.xls"
        return dest, raw_url, "xls"

    # 7. Archives
    if mime_type == "application/zip":
        dest = DOWNLOADS_DIR / "archives" / f"{doc_id}.zip"
        return dest, raw_url, "zip"
    if mime_type == "application/vnd.rar":
        dest = DOWNLOADS_DIR / "archives" / f"{doc_id}.rar"
        return dest, raw_url, "rar"

    # 8. Images
    if mime_type == "image/jpeg":
        dest = DOWNLOADS_DIR / "images" / f"{doc_id}.jpg"
        return dest, raw_url, "jpg"
    if mime_type == "image/png":
        dest = DOWNLOADS_DIR / "images" / f"{doc_id}.png"
        return dest, raw_url, "png"

    # 9. Media
    if mime_type == "video/mp4":
        dest = DOWNLOADS_DIR / "media" / f"{doc_id}.mp4"
        return dest, raw_url, "mp4"
    if mime_type == "video/webm":
        dest = DOWNLOADS_DIR / "media" / f"{doc_id}.webm"
        return dest, raw_url, "webm"

    # 10. Text / Misc
    if mime_type == "text/plain":
        dest = DOWNLOADS_DIR / "text" / f"{doc_id}.txt"
        return dest, raw_url, "txt"

    # Fallback octet-stream (handle Google Drive files properly)
    dest = DOWNLOADS_DIR / "other" / f"{doc_id}.bin"
    if fid and ("drive.google.com" in raw_url or "docs.google.com" in raw_url):
        dl_url = f"https://drive.google.com/uc?id={fid}&export=download"
    else:
        dl_url = raw_url
    return dest, dl_url, "binary"


def get_drive_confirm_token(html: str) -> str | None:
    """Extract confirm token for Google Drive downloads."""
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
    mime_type: str,
    raw_url: str,
    title: str,
    force: bool,
) -> dict:
    dest_path, dl_url, category = get_target_path_and_url(doc_id, mime_type, raw_url)

    # Check cache
    if not force and dest_path.exists() and dest_path.stat().st_size > 100:
        return {
            "doc_id": doc_id,
            "success": True,
            "status": "cached",
            "size": dest_path.stat().st_size,
            "local_path": str(dest_path),
            "category": category,
        }

    is_google = "google.com" in dl_url
    is_dropbox = "dropbox.com" in dl_url
    active_sem = drive_sem if is_google else general_sem

    # Prepare URL for Dropbox
    if is_dropbox:
        dl_url = re.sub(r"[?&]dl=0", "", dl_url)
        sep = "&" if "?" in dl_url else "?"
        dl_url = f"{dl_url}{sep}dl=1"

    async with active_sem:
        max_attempts = 3 if is_google else 2
        for attempt in range(1, max_attempts + 1):
            try:
                if is_google:
                    await asyncio.sleep(0.25)

                resp = await client.get(dl_url)

                # Google Drive "Can't download file" permission check
                if is_google and (b"Can't download file" in resp.content or b"permission to download this file" in resp.content):
                    return {"doc_id": doc_id, "success": False, "status": "download_disabled", "category": category}

                # Fail fast on 404 / 401 / 403
                if resp.status_code in (404, 401, 403):
                    return {"doc_id": doc_id, "success": False, "status": f"http_{resp.status_code}", "category": category}

                # Google login redirect
                if is_google and "accounts.google.com" in str(resp.url):
                    if attempt < max_attempts:
                        await asyncio.sleep(2.0 * attempt)
                        continue
                    return {"doc_id": doc_id, "success": False, "status": "auth_wall", "category": category}

                # Google large file virus warning
                if resp.status_code == 200 and is_google and "export" in dl_url and len(resp.content) < 3000:
                    token = get_drive_confirm_token(resp.text)
                    if token:
                        sep = "&" if "?" in dl_url else "?"
                        confirm_url = f"{dl_url}{sep}confirm={token}"
                        resp = await client.get(confirm_url)

                if resp.status_code == 429:
                    if attempt < max_attempts:
                        await asyncio.sleep(3.0 * attempt)
                        continue
                    return {"doc_id": doc_id, "success": False, "status": "rate_limited", "category": category}

                if resp.status_code != 200:
                    return {"doc_id": doc_id, "success": False, "status": f"http_{resp.status_code}", "category": category}

                content = resp.content
                if len(content) == 0:
                    return {"doc_id": doc_id, "success": False, "status": "empty_content", "category": category}

                # Write to disk
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with dest_path.open("wb") as f:
                    f.write(content)

                return {
                    "doc_id": doc_id,
                    "success": True,
                    "status": "downloaded",
                    "size": len(content),
                    "local_path": str(dest_path),
                    "category": category,
                }

            except httpx.TimeoutException:
                if attempt < max_attempts:
                    await asyncio.sleep(1.5)
                    continue
                return {"doc_id": doc_id, "success": False, "status": "timeout", "category": category}
            except Exception as e:
                if attempt < max_attempts:
                    await asyncio.sleep(1.5)
                    continue
                return {"doc_id": doc_id, "success": False, "status": "error", "error": str(e), "category": category}

    return {"doc_id": doc_id, "success": False, "status": "failed", "category": category}


async def run_pipeline(
    db_path: str = "stash.db",
    force: bool = False,
    mime_filter: str | None = None,
    limit: int | None = None,
) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = """
    SELECT doc_id, type, url, title, local_path, status
    FROM document_stubs
    WHERE type != 'application/pdf'
    """
    params = []
    if mime_filter:
        query += " AND (type = ? OR file_type = ?)"
        params.extend([mime_filter, mime_filter])
    query += " ORDER BY doc_id ASC"

    rows = cur.execute(query, params).fetchall()
    if limit:
        rows = rows[:limit]

    total = len(rows)
    logger.info("Found %d non-PDF document stubs to process...", total)
    if total == 0:
        logger.info("No non-PDF stubs match filter.")
        conn.close()
        return

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=12)
    timeout = httpx.Timeout(30.0, connect=10.0)

    general_sem = asyncio.Semaphore(8)
    drive_sem = asyncio.Semaphore(4)

    stats = {"downloaded": 0, "cached": 0, "failed": 0, "bytes": 0}
    by_category = {}
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
            fetch_one(client, general_sem, drive_sem, r_id, r_type, r_url, r_title, force)
            for r_id, r_type, r_url, r_title, _, _ in rows
        ]

        batch_updates = []
        for fut in asyncio.as_completed(tasks):
            res = await fut
            completed += 1
            cat = res.get("category", "other")
            by_category[cat] = by_category.get(cat, 0) + 1

            if res["success"]:
                if res["status"] == "cached":
                    stats["cached"] += 1
                else:
                    stats["downloaded"] += 1
                    stats["bytes"] += res.get("size", 0)
            else:
                stats["failed"] += 1

            batch_updates.append(res)

            # Flush batch to SQLite periodically
            if len(batch_updates) >= 20 or completed == total:
                for item in batch_updates:
                    d_id = item["doc_id"]
                    if item["success"]:
                        lp = item["local_path"]
                        cur.execute(
                            """
                            UPDATE document_stubs
                            SET local_path = ?,
                                status = CASE WHEN status = 'unavailable' THEN 'downloaded' ELSE status END
                            WHERE doc_id = ?
                            """,
                            (lp, d_id),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE document_stubs
                            SET status = 'unavailable'
                            WHERE doc_id = ? AND status NOT IN ('extracted', 'downloaded')
                            """,
                            (d_id,),
                        )

                conn.commit()
                batch_updates.clear()

            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / max(0.1, elapsed)
                mb = stats["bytes"] / (1024 * 1024)
                logger.info(
                    "Progress: %d/%d (%.1f%%) | Downloaded: %d (%.1f MB) | Cached: %d | Failed/Unavail: %d | Rate: %.1f files/s",
                    completed,
                    total,
                    completed / total * 100,
                    stats["downloaded"],
                    mb,
                    stats["cached"],
                    stats["failed"],
                    rate,
                )

    conn.close()
    elapsed = time.time() - t0
    logger.info("=== Non-PDF Document Fetching Completed in %.1fs ===", elapsed)
    logger.info("  Newly Downloaded : %d (%.1f MB)", stats["downloaded"], stats["bytes"] / (1024 * 1024))
    logger.info("  Already Cached   : %d", stats["cached"])
    logger.info("  Failed / Unavail : %d", stats["failed"])
    logger.info("--- By Category ---")
    for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
        logger.info("  %-20s : %d", cat, count)


def main():
    parser = argparse.ArgumentParser(description="Download all non-PDF document stubs (Word, Excel, Docs, Sheets, etc.)")
    parser.add_argument("--db", default="stash.db", help="Path to database")
    parser.add_argument("--force", action="store_true", help="Force re-download cached files")
    parser.add_argument("--type", help="Filter by MIME type or file_type")
    parser.add_argument("--limit", type=int, help="Limit number of files to fetch")
    args = parser.parse_args()

    asyncio.run(run_pipeline(db_path=args.db, force=args.force, mime_filter=args.type, limit=args.limit))


if __name__ == "__main__":
    main()
