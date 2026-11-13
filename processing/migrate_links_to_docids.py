"""
Script to migrate internal document and attachment links to canonical `/docs/{doc_id}` references.

- Rewrites markdown links pointing to crawled FTU pages to `/docs/{existing_id}`.
- Discovers external document assets (PDFs, DOCX, Google Docs/Sheets/Drive/Forms) linked in markdown.
- Inserts clean stub records into `documents` for each unique document asset.
- Rewrites the markdown links pointing to those assets to `/docs/{new_id}`.
- Leaves images alone inline (including [image file] and image file extensions).
- Leaves social media links untouched.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_links")

SOCIAL_DOMAINS = {
    "facebook.com", "m.facebook.com", "l.facebook.com", "fb.me", "fb.com",
    "youtube.com", "youtu.be", "instagram.com", "linkedin.com",
    "twitter.com", "x.com", "tiktok.com", "zalo.me", "threads.net",
    "pinterest.com",
}

GENERIC_TITLES = {
    "", "tại đây", "tại đây.", "link", "click here", "click here!",
    "xem tại đây", "tải về", "xem chi tiết", "đây", "xem", "download",
}

IMG_EXT_REGEX = re.compile(r"\.(jpg|jpeg|png|gif|webp|svg|ico|bmp)(\?.*)?$", re.IGNORECASE)
DOC_EXT_REGEX = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|rar|zip)(\?.*)?$", re.IGNORECASE)
MD_LINK_REGEX = re.compile(r'\[([^\]]+)\]\((https?://[^\s\"\)]+)(?:\s+\"([^\"]*)\")?\)')


def normalize_url(u: str) -> str:
    """Normalize URL for consistent lookup."""
    if not u:
        return ""
    u = unquote(u.strip())
    u = u.replace(chr(92), "/")  # normalize backslashes
    p = urlparse(u)
    scheme = "https" if p.scheme in ("http", "https") else p.scheme
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = p.path.rstrip("/")
    if "google.com" in netloc:
        query = p.query
        return f"{scheme}://{netloc}{path}?{query}" if query else f"{scheme}://{netloc}{path}"
    return f"{scheme}://{netloc}{path}"


def is_social(url_lower: str) -> bool:
    return any(domain in url_lower for domain in SOCIAL_DOMAINS)


def is_image(url_lower: str) -> bool:
    return bool(IMG_EXT_REGEX.search(url_lower))


def is_pdf_or_office(url_lower: str) -> bool:
    return bool(DOC_EXT_REGEX.search(url_lower))


def is_google_docs(url_lower: str) -> bool:
    return "drive.google.com" in url_lower or "docs.google.com" in url_lower


def generate_fallback_title(url: str, current_title: str, parent_id: int) -> str:
    """Generate a meaningful title for a document if the anchor text is generic."""
    cleaned = (current_title or "").strip()
    if cleaned and cleaned.lower() not in GENERIC_TITLES:
        return cleaned

    u_lower = url.lower()
    p = urlparse(url)
    last_seg = p.path.split("/")[-1].split("?")[0] if p.path else ""
    if last_seg and "." in last_seg:
        return unquote(last_seg)
    if "forms" in u_lower:
        return f"Biểu mẫu Google Form (tham chiếu từ doc #{parent_id})"
    if "spreadsheets" in u_lower:
        return f"Bảng tính Google Sheets (tham chiếu từ doc #{parent_id})"
    if "document" in u_lower:
        return f"Tài liệu Google Docs (tham chiếu từ doc #{parent_id})"
    if "drive" in u_lower or "file" in u_lower:
        return f"Tệp Google Drive (tham chiếu từ doc #{parent_id})"
    return f"Tài liệu tham chiếu #{parent_id}"


def run_migration(db_path: str = "stash.db", dry_run: bool = False) -> None:
    logger.info("Opening database at %s (dry_run=%s)", db_path, dry_run)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Build map of existing URLs
    cursor.execute("SELECT id, url, redirected_url FROM documents")
    existing_urls: dict[str, int] = {}
    for doc_id, u, ru in cursor.fetchall():
        if u:
            norm = normalize_url(u)
            if norm:
                existing_urls[norm] = doc_id
        if ru:
            norm_ru = normalize_url(ru)
            if norm_ru:
                existing_urls[norm_ru] = doc_id

    logger.info("Loaded %d existing normalized URLs from database.", len(existing_urls))

    # 2. Collect markdown links needing stubs (PDFs & Google Docs/Drive)
    cursor.execute("SELECT id, markdown FROM documents WHERE markdown IS NOT NULL")
    rows = cursor.fetchall()
    logger.info("Scanning %d documents for links...", len(rows))

    to_stub: dict[str, dict] = {}
    for doc_id, md in rows:
        for m in MD_LINK_REGEX.finditer(md):
            text = m.group(1).strip()
            href = m.group(2).strip()
            href_l = href.lower()

            if is_image(href_l) or is_social(href_l):
                continue

            norm = normalize_url(href)
            if norm in existing_urls:
                continue

            if is_pdf_or_office(href_l):
                ftype = "pdf_or_office"
            elif is_google_docs(href_l):
                ftype = "google_docs"
            else:
                continue

            if norm not in to_stub:
                to_stub[norm] = {
                    "url": href,
                    "title": text,
                    "parent_id": doc_id,
                    "type": ftype,
                }
            else:
                old_title = to_stub[norm]["title"]
                if old_title.lower() in GENERIC_TITLES and text.lower() not in GENERIC_TITLES:
                    to_stub[norm]["title"] = text
                elif len(text) > len(old_title) and not text.lower().startswith("http") and text.lower() not in GENERIC_TITLES:
                    to_stub[norm]["title"] = text

    logger.info("Identified %d unique document assets to stub.", len(to_stub))

    # 3. Insert stubs into documents
    new_stubs_count = 0
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    if not dry_run:
        for norm, item in to_stub.items():
            final_title = generate_fallback_title(item["url"], item["title"], item["parent_id"])
            file_type = item["type"]
            raw_url = item["url"]
            parent_id = item["parent_id"]

            stub_md = (
                f"# {final_title}\n\n"
                f"- **Nguồn tài liệu**: [{raw_url}]({raw_url})\n"
                f"- **Tài liệu tham chiếu từ bài viết**: [/docs/{parent_id}](/docs/{parent_id})\n"
                f"- **Loại tệp**: {file_type}\n\n"
                f"*Chờ trích xuất văn bản chi tiết và tạo vector embedding.*"
            )

            metadata_json = json.dumps({
                "title": final_title,
                "source": file_type,
                "parent_doc_id": parent_id,
                "status": "pending_text_extraction",
            }, ensure_ascii=False)

            cursor.execute(
                """
                INSERT INTO documents (
                    url, html, success, media, links, tables,
                    markdown, metadata, saved_at
                ) VALUES (?, '', 1, '{}', '{}', '[]', ?, ?, ?)
                """,
                (raw_url, stub_md, metadata_json, now_str)
            )
            new_id = cursor.lastrowid
            existing_urls[norm] = new_id
            new_stubs_count += 1

        logger.info("Successfully inserted %d stub documents into documents.", new_stubs_count)

    # 4. Rewrite markdown links across all documents
    total_rewrites = 0
    updated_docs = 0

    def link_replacer(match: re.Match) -> str:
        nonlocal total_rewrites
        text = match.group(1)
        href = match.group(2)
        title = match.group(3)
        href_l = href.lower()

        # Skip inline images or anchors with 'image file'
        if "image file" in text.lower() or is_image(href_l):
            return match.group(0)

        # Skip social media
        if is_social(href_l):
            return match.group(0)

        norm = normalize_url(href)
        if norm in existing_urls:
            target_id = existing_urls[norm]
            total_rewrites += 1
            title_part = f' "{title}"' if title else ""
            return f"[{text}](/docs/{target_id}{title_part})"

        return match.group(0)

    # Re-fetch all documents including newly inserted stubs
    cursor.execute("SELECT id, markdown FROM documents WHERE markdown IS NOT NULL")
    all_docs = cursor.fetchall()

    for doc_id, md in all_docs:
        new_md = MD_LINK_REGEX.sub(link_replacer, md)
        if new_md != md:
            updated_docs += 1
            if not dry_run:
                cursor.execute(
                    "UPDATE documents SET markdown = ? WHERE id = ?",
                    (new_md, doc_id)
                )

    if not dry_run:
        conn.commit()
        logger.info("Changes committed to database successfully.")
    else:
        logger.info("Dry run finished; no changes were written.")

    conn.close()

    logger.info("=== Migration Summary ===")
    logger.info("New document stubs created: %d", new_stubs_count)
    logger.info("Documents with rewritten links: %d", updated_docs)
    logger.info("Total link occurrences rewritten to /docs/{id}: %d", total_rewrites)


if __name__ == "__main__":
    is_dry = "--dry-run" in sys.argv
    run_migration(dry_run=is_dry)
