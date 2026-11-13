"""
Populates `document_stubs` table in `stash.db` to isolate all document stubs
(PDFs, Word docs, Google Docs/Sheets/Drive/Forms, Dropbox files) for downstream
text extraction and embedding.

Also discovers any remaining un-stubbed document links (e.g. empty-anchor links
in tables) and updates their markdown references to `/docs/{new_doc_id}`.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("populate_stubs")

DOC_EXT_REGEX = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|rar|zip|csv|txt)(\?.*)?$", re.IGNORECASE)
MD_LINK_ANY = re.compile(r'\[([^\]]*)\]\((https?://[^\s\"\)]+)(?:\s+\"([^\"]*)\")?\)')

GENERIC_TITLES = {
    "", "tại đây", "tại đây.", "link", "click here", "click here!",
    "xem tại đây", "tải về", "xem chi tiết", "đây", "xem", "download",
}


def normalize_url(u: str) -> str:
    if not u:
        return ""
    u = unquote(u.strip()).replace(chr(92), "/")
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


def clean_slug_title(url: str, current_title: str, parent_id: int = 0) -> str:
    cleaned = (current_title or "").strip()
    if cleaned and cleaned.lower() not in GENERIC_TITLES:
        return cleaned

    p = urlparse(url)
    last_seg = p.path.split("/")[-1].split("?")[0] if p.path else ""
    last_seg = unquote(last_seg)
    # Strip numeric timestamp hash
    subbed = re.sub(r"-(?:doc|docx|pdf)?-\d{6,12}(\.[a-zA-Z0-9]+)$", r"\1", last_seg, flags=re.IGNORECASE)
    subbed = re.sub(r"-\d{6,12}(\.[a-zA-Z0-9]+)$", r"\1", subbed)
    base = subbed.rsplit(".", 1)[0]
    ext = subbed.rsplit(".", 1)[1] if "." in subbed else ""
    readable = base.replace("-", " ").replace("_", " ").strip()
    readable = re.sub(r"\s+", " ", readable)
    if readable:
        return f"{readable.title()} ({ext.upper()})" if ext else readable.title()
    if "forms" in url.lower():
        return f"Biểu mẫu Google Form (#{parent_id})"
    if "spreadsheets" in url.lower():
        return f"Bảng tính Google Sheets (#{parent_id})"
    if "drive" in url.lower():
        return f"Tệp Google Drive (#{parent_id})"
    return f"Tài liệu tham chiếu (#{parent_id})"


def get_file_type(url: str) -> str:
    u_lower = url.lower()
    m = DOC_EXT_REGEX.search(u_lower)
    if m:
        return m.group(1).lower()
    if "forms" in u_lower:
        return "google_form"
    if "spreadsheets" in u_lower:
        return "google_sheet"
    if "document" in u_lower:
        return "google_doc"
    if "drive" in u_lower or "docs.google.com" in u_lower:
        return "google_drive"
    return "document"


def main():
    conn = sqlite3.connect("stash.db")
    cursor = conn.cursor()

    # Ensure document_stubs table exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_stubs (
        doc_id INTEGER PRIMARY KEY,
        url TEXT NOT NULL UNIQUE,
        title TEXT NOT NULL,
        file_type TEXT NOT NULL,
        source_domain TEXT NOT NULL,
        parent_doc_id INTEGER,
        status TEXT DEFAULT 'pending',
        local_path TEXT,
        extracted_chars INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_stubs_status ON document_stubs(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_stubs_type ON document_stubs(file_type);")

    # 1. Existing stubs created in previous pass (id >= 8392)
    cursor.execute("SELECT id, url, metadata FROM documents WHERE id >= 8392")
    stubs_8392 = cursor.fetchall()
    for doc_id, url, meta_raw in stubs_8392:
        meta = json.loads(meta_raw) if meta_raw else {}
        title = meta.get("title") or clean_slug_title(url, "")
        ftype = get_file_type(url)
        domain = urlparse(url).netloc
        parent_id = meta.get("parent_doc_id")
        cursor.execute("""
        INSERT OR IGNORE INTO document_stubs (doc_id, url, title, file_type, source_domain, parent_doc_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, url, title, ftype, domain, parent_id))

    # 2. Existing 0-byte PDFs/Docs from original crawl
    cursor.execute("""
    SELECT id, url, metadata FROM documents
    WHERE id < 8392 AND length(markdown) = 0
      AND (url LIKE '%.pdf%' OR url LIKE '%.doc%' OR url LIKE '%.docx%' OR url LIKE '%.xls%')
    """)
    zero_bytes = cursor.fetchall()
    for doc_id, url, meta_raw in zero_bytes:
        meta = json.loads(meta_raw) if meta_raw else {}
        title = meta.get("title") or clean_slug_title(url, "", doc_id)
        ftype = get_file_type(url)
        domain = urlparse(url).netloc
        cursor.execute("""
        INSERT OR IGNORE INTO document_stubs (doc_id, url, title, file_type, source_domain, parent_doc_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, url, title, ftype, domain, None))

        # Update markdown with clean stub text so /docs/{id} is informative
        stub_md = (
            f"# {title}\n\n"
            f"- **Nguồn tài liệu**: [{url}]({url})\n"
            f"- **Định dạng**: {ftype}\n\n"
            f"*Tài liệu stub đang chờ tải xuống, trích xuất văn bản và tạo vector embedding.*"
        )
        cursor.execute("UPDATE documents SET markdown = ? WHERE id = ?", (stub_md, doc_id))

    logger.info("Indexed existing stubs: %d step-1 stubs + %d 0-byte crawl documents.", len(stubs_8392), len(zero_bytes))

    # 3. Discover un-stubbed document links in markdown (including empty anchors []())
    existing_urls = {}
    cursor.execute("SELECT id, url, redirected_url FROM documents")
    for doc_id, u, ru in cursor.fetchall():
        if u:
            existing_urls[normalize_url(u)] = doc_id
        if ru:
            existing_urls[normalize_url(ru)] = doc_id

    cursor.execute("SELECT id, markdown FROM documents WHERE markdown IS NOT NULL")
    docs = cursor.fetchall()

    new_stubs = {}
    for doc_id, md in docs:
        for m in MD_LINK_ANY.finditer(md):
            text = m.group(1).strip()
            href = m.group(2).strip()
            href_l = href.lower()
            norm = normalize_url(href)

            if norm in existing_urls:
                continue

            # Check if document / office file
            is_doc = bool(DOC_EXT_REGEX.search(href_l)) or any(d in href_l for d in ["drive.google.com", "docs.google.com", "dropbox.com"])
            if not is_doc:
                continue

            if norm not in new_stubs or (len(text) > len(new_stubs[norm]["title"])):
                new_stubs[norm] = {"url": href, "title": text, "parent_id": doc_id}

    logger.info("Found %d un-stubbed document links in markdown to create.", len(new_stubs))

    # Insert new stubs
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for norm, item in new_stubs.items():
        raw_url = item["url"]
        parent_id = item["parent_id"]
        title = clean_slug_title(raw_url, item["title"], parent_id)
        ftype = get_file_type(raw_url)
        domain = urlparse(raw_url).netloc

        stub_md = (
            f"# {title}\n\n"
            f"- **Nguồn tài liệu**: [{raw_url}]({raw_url})\n"
            f"- **Tài liệu tham chiếu từ bài viết**: [/docs/{parent_id}](/docs/{parent_id})\n"
            f"- **Định dạng**: {ftype}\n\n"
            f"*Tài liệu stub đang chờ tải xuống, trích xuất văn bản và tạo vector embedding.*"
        )
        meta_json = json.dumps({
            "title": title,
            "source": ftype,
            "parent_doc_id": parent_id,
            "status": "pending_text_extraction",
        }, ensure_ascii=False)

        cursor.execute("""
        INSERT INTO documents (url, html, success, media, links, tables, markdown, metadata, cache_status, saved_at)
        VALUES (?, '', 1, '{}', '{}', '[]', ?, ?, 'stub', ?)
        """, (raw_url, stub_md, meta_json, now_str))

        new_doc_id = cursor.lastrowid
        existing_urls[norm] = new_doc_id

        cursor.execute("""
        INSERT OR IGNORE INTO document_stubs (doc_id, url, title, file_type, source_domain, parent_doc_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (new_doc_id, raw_url, title, ftype, domain, parent_id))

    # 4. Rewrite markdown links for any documents containing these new targets
    updated_count = 0
    for doc_id, md in docs:
        def replace_link(match):
            text = match.group(1).strip()
            href = match.group(2).strip()
            title_attr = match.group(3)
            norm = normalize_url(href)
            if norm in existing_urls:
                target_id = existing_urls[norm]
                display_text = text if text else clean_slug_title(href, "", doc_id)
                t_str = f' "{title_attr}"' if title_attr else ""
                return f"[{display_text}](/docs/{target_id}{t_str})"
            return match.group(0)

        new_md = MD_LINK_ANY.sub(replace_link, md)
        if new_md != md:
            updated_count += 1
            cursor.execute("UPDATE documents SET markdown = ? WHERE id = ?", (new_md, doc_id))

    conn.commit()

    # Final tally
    cursor.execute("SELECT count(*) FROM document_stubs")
    total_stubs = cursor.fetchone()[0]
    cursor.execute("SELECT file_type, count(*) FROM document_stubs GROUP BY 1 ORDER BY 2 DESC")
    breakdown = cursor.fetchall()

    logger.info("=== Isolation Complete ===")
    logger.info("Total isolated stubs in `document_stubs`: %d", total_stubs)
    for ftype, cnt in breakdown:
        logger.info("  %-15s : %d", ftype, cnt)
    logger.info("Documents updated with new link targets: %d", updated_count)

    conn.close()


if __name__ == "__main__":
    main()
