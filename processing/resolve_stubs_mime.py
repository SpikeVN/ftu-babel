"""
Resolve real MIME types for all document stubs using HTTP HEAD (and streaming GET fallback).

Populates:
- `document_stubs.mime_type`
- `document_stubs.type`
- `document_stubs.title` (if real filename discovered from Content-Disposition)
- `documents.metadata.mime_type`
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import sqlite3
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("resolve_mime")

# Common document extension to MIME type map
MIME_MAP = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "ppt": "application/vnd.ms-powerpoint",
    "zip": "application/zip",
    "rar": "application/vnd.rar",
    "7z": "application/x-7z-compressed",
    "txt": "text/plain",
    "csv": "text/csv",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "mp4": "video/mp4",
}

GENERIC_TITLES = {
    "", "tại đây", "tại đây.", "link", "click here", "click here!",
    "xem tại đây", "tải về", "xem chi tiết", "đây", "xem", "download",
}


def parse_filename_from_cd(cd: str) -> str:
    """Extract filename from Content-Disposition header."""
    if not cd:
        return ""
    # Try filename*
    m_star = re.search(r"filename\*=(?:UTF-8|utf-8)''([^;]+)", cd)
    if m_star:
        return unquote(m_star.group(1)).strip().strip('"\'')
    # Try filename=
    m_plain = re.search(r'filename="?([^";]+)"?', cd)
    if m_plain:
        return unquote(m_plain.group(1)).strip().strip('"\'')
    return ""


def mime_from_filename(filename: str) -> str:
    """Infer MIME type from filename extension."""
    if not filename:
        return ""
    m = re.search(r"\.([a-zA-Z0-9]{2,5})$", filename.lower())
    if m:
        ext = m.group(1)
        if ext in MIME_MAP:
            return MIME_MAP[ext]
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
    return ""


async def resolve_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    doc_id: int,
    url: str,
    title: str,
    domain: str,
) -> tuple[int, str, str | None]:
    """
    Resolves the MIME type and optional filename for a single stub.
    Returns (doc_id, mime_type, updated_title).
    """
    async with semaphore:
        u_lower = url.lower()
        p = urlparse(url)
        path = p.path.lower()

        # 1. Native Google Workspace web endpoints
        if "google.com" in domain:
            if "/document/" in path:
                return doc_id, "application/vnd.google-apps.document", None
            if "/spreadsheets/" in path:
                return doc_id, "application/vnd.google-apps.spreadsheet", None
            if "/forms/" in path:
                return doc_id, "application/vnd.google-apps.form", None
            if "/presentation/" in path:
                return doc_id, "application/vnd.google-apps.presentation", None
            if "/folders/" in path:
                return doc_id, "application/vnd.google-apps.folder", None

            # Google Drive files (/file/d/ or open?id=)
            file_id = ""
            m_file = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
            m_open = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
            if m_file:
                file_id = m_file.group(1)
            elif m_open:
                file_id = m_open.group(1)

            if file_id:
                uc_url = f"https://drive.google.com/uc?id={file_id}&export=download"
                try:
                    r = await client.head(uc_url)
                    cd = r.headers.get("Content-Disposition", "")
                    ct = r.headers.get("Content-Type", "").split(";")[0].strip()

                    fn = parse_filename_from_cd(cd)
                    inferred = mime_from_filename(fn)
                    if inferred:
                        new_title = fn if (not title or title.strip().lower() in GENERIC_TITLES) else None
                        return doc_id, inferred, new_title

                    if ct and ct not in ("text/html", "application/octet-stream"):
                        return doc_id, ct, None

                    # If server returned octet-stream, check title for extension
                    title_mime = mime_from_filename(title)
                    if title_mime:
                        return doc_id, title_mime, None

                    return doc_id, "application/pdf", None  # Standard fallback for Google Drive PDFs
                except Exception as e:
                    logger.debug("Doc #%d Google Drive error: %s", doc_id, e)
                    return doc_id, "application/octet-stream", None

        # 2. Dropbox links: ensure dl=1 to fetch binary headers
        target_url = url
        if "dropbox.com" in domain and "dl=0" in target_url:
            target_url = target_url.replace("dl=0", "dl=1")

        # 3. Direct URLs (FTU, Dropbox, other servers)
        try:
            r = await client.head(target_url)
            # If 405 Method Not Allowed or 403 on HEAD, fallback to GET stream
            if r.status_code in (403, 405):
                async with client.stream("GET", target_url) as stream_resp:
                    cd = stream_resp.headers.get("Content-Disposition", "")
                    ct = stream_resp.headers.get("Content-Type", "").split(";")[0].strip()
            else:
                cd = r.headers.get("Content-Disposition", "")
                ct = r.headers.get("Content-Type", "").split(";")[0].strip()

            fn = parse_filename_from_cd(cd)
            inferred = mime_from_filename(fn) or mime_from_filename(url)
            if inferred and (ct in ("text/html", "application/octet-stream", "application/json", "") or not ct.startswith("application/")):
                new_title = fn if (not title or title.strip().lower() in GENERIC_TITLES) else None
                return doc_id, inferred, new_title

            if ct and ct != "text/html":
                new_title = fn if (fn and (not title or title.strip().lower() in GENERIC_TITLES)) else None
                return doc_id, ct, new_title

            if inferred:
                return doc_id, inferred, None

            return doc_id, "application/octet-stream", None
        except Exception as e:
            logger.debug("Doc #%d request error: %s", doc_id, e)
            # Fallback based on URL extension
            inferred = mime_from_filename(url)
            return doc_id, inferred or "application/octet-stream", None


async def main_async(db_path: str = "stash.db", concurrency: int = 30):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT doc_id, url, title, source_domain FROM document_stubs ORDER BY doc_id ASC")
    rows = cursor.fetchall()
    total = len(rows)
    logger.info("Found %d stubs to resolve MIME types (concurrency=%d)...", total, concurrency)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    limits = httpx.Limits(max_keepalive_connections=concurrency * 2, max_connections=concurrency)
    timeout = httpx.Timeout(10.0, connect=5.0)

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with httpx.AsyncClient(verify=False, follow_redirects=True, headers=headers, limits=limits, timeout=timeout) as client:
        tasks = [
            resolve_one(client, semaphore, doc_id, url, title, domain)
            for doc_id, url, title, domain in rows
        ]

        completed = 0
        for fut in asyncio.as_completed(tasks):
            res = await fut
            results.append(res)
            completed += 1
            if completed % 200 == 0 or completed == total:
                logger.info("Progress: %d / %d (%.1f%%)", completed, total, completed / total * 100)

    logger.info("Updating database with resolved MIME types...")
    # Update document_stubs
    for doc_id, mime_type, new_title in results:
        if new_title:
            cursor.execute("""
            UPDATE document_stubs
            SET mime_type = ?, type = ?, title = ?
            WHERE doc_id = ?
            """, (mime_type, mime_type, new_title, doc_id))
        else:
            cursor.execute("""
            UPDATE document_stubs
            SET mime_type = ?, type = ?
            WHERE doc_id = ?
            """, (mime_type, mime_type, doc_id))

        # Update documents metadata
        cursor.execute("""
        UPDATE documents
        SET metadata = json_set(COALESCE(metadata, '{}'), '$.mime_type', ?)
        WHERE id = ?
        """, (mime_type, doc_id))

    conn.commit()

    # Breakdown report
    cursor.execute("SELECT type, count(*) FROM document_stubs GROUP BY 1 ORDER BY 2 DESC")
    breakdown = cursor.fetchall()

    logger.info("=== MIME Type Resolution Summary ===")
    for mtype, cnt in breakdown:
        logger.info("  %-65s : %d", mtype, cnt)

    conn.close()


if __name__ == "__main__":
    asyncio.run(main_async())
