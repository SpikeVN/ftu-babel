"""
process_cleanup.py - HTML & Markdown Cleanup Pipeline for stash.db.

Cleans up raw HTML crawled content, converts it into clean Markdown, purges
unwanted sections (Sidebar, Đào tạo, E-Learning, Quy định, Kết nối), replaces
inline images with [image file], strips out website footers/boilerplates, and
saves the clean Markdown into the 'markdown' column of stash.db.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import html2text
from tqdm import tqdm

try:
    from server.config import DEFAULT_DB_PATH
except ImportError:
    DEFAULT_DB_PATH = "stash.db"

logger = logging.getLogger(__name__)

# Headers to purge from markdown (title, heading level (0 = any level 1-6), instance)
HEADERS_TO_PURGE: list[tuple[str, int, str]] = [
    ("Sidebar", 0, "first"),
    ("Đào tạo", 0, "last"),
    ("E-Learning", 0, "last"),
    ("Quy định", 0, "last"),
    ("Kết nối", 0, "last"),
]

# Regexes for markdown and HTML cleanup
REGEX_REMOVE_MARKDOWN_IMAGE = re.compile(r"!\[(?:[^\[\]]|\[[^\]]*\])*\]\([^\)]*\)")
REGEX_REMOVE_MARKDOWN_LINKS = re.compile(r"\[(?:[^\[\]]|\[[^\]]*\])*\]\([^\)]*\)")
REGEX_STRIP_MARQUEE = re.compile(r"<marquee[\s\S]*?</marquee>", re.IGNORECASE)
REGEX_STRIP_SCRIPTS = re.compile(r"<(script|style)[\s\S]*?</\1>", re.IGNORECASE)
REGEX_TOP_NAV = re.compile(
    r"^(?:\s*[\*\-]\s+\[.*?\]\(.*?\)\s*\n|\s*Tìm kiếm\.\.\.\s*\n|\s*__\s*\n|\s*\n)+(?=#)",
    re.MULTILINE,
)
# Strip top header navigation (FTU, EN, Bộ Giáo dục, Kỷ niệm, Liên hệ, Tìm kiếm)
REGEX_HEADER_NAV = re.compile(
    r"^\s*\* \[FTU\]\(https?://(?:www\.)?ftu\.edu\.vn/?\).*?\[Liên hệ\]\([^\)]+\)(?:\s*__)?(?:\s*Tìm kiếm\.\.\.)?\s*",
    re.DOTALL,
)

# Standard FTU / Teline V Joomla template footer boilerplate
FOOTER_BOILERPLATE = """[ [image file] Teline V ](https://ftu.edu.vn/ftu "Teline V") Best News Template For Joomla
Trường Đại học Ngoại thương   
______________________________________  
Thành lập: 1960  
______________________________________  
Loại hình: Đại học công lập  
______________________________________  
Địa chỉ: Số 91 Phố Chùa Láng, Phường Láng, Hà Nội, Việt Nam  
______________________________________
Điện thoại: (84-24) 3259 5158  
______________________________________"""

# Compile header purge patterns
COMPILED_PURGE_LIST: list[tuple[re.Pattern, str]] = []
for title, level, instance in HEADERS_TO_PURGE:
    escaped_title = re.escape(title)
    if level == 0:
        raw_pattern = (
            rf"^(#{{1,6}})\s+{escaped_title}\s*\n"
            rf"(?:(?!^#{{1,6}}\s+).|\n)*?"
            rf"(?=\Z|^#{{1,6}}\s+)"
        )
    else:
        hashes = "#" * level
        raw_pattern = (
            rf"^{hashes}\s+{escaped_title}\s*\n"
            rf"[\s\S]*?"
            rf"(?=\Z|^#{{1,{level}}}\s+)"
        )
    COMPILED_PURGE_LIST.append((re.compile(raw_pattern, flags=re.MULTILINE), instance))


def purge_section(text: str, regex: re.Pattern, instance: str, wildcard: bool) -> str:
    """Purge a matched section from the markdown text."""
    matches = list(regex.finditer(text))
    if not matches:
        return text

    target_match = matches[0] if instance == "first" else matches[-1]

    if wildcard:
        target_hashes = len(target_match.group(1))
        start = target_match.start()
        remaining_text = text[target_match.end():]

        # Look for the next header with target_hashes or fewer (#s)
        boundary_pattern = rf"^#{{1,{target_hashes}}}\s+"
        boundary_match = re.search(boundary_pattern, remaining_text, re.MULTILINE)

        if boundary_match:
            end = target_match.end() + boundary_match.start()
        else:
            end = len(text)
        return text[:start] + text[end:]
    else:
        start, end = target_match.span()
        return text[:start] + text[end:]


def clean_markdown(doc: str) -> str:
    """Purge all configured sections (Sidebar, Đào tạo, E-Learning, etc.) from markdown."""
    for (compiled_regex, instance), (_, level, _) in zip(COMPILED_PURGE_LIST, HEADERS_TO_PURGE):
        doc = purge_section(doc, compiled_regex, instance, wildcard=(level == 0))
    return doc


def strip_header_nav(text: str) -> str:
    """Strip FTU top navigation menu (FTU, EN, BGD, Kỷ niệm, Liên hệ, etc.)."""
    if not text:
        return ""
    text = REGEX_HEADER_NAV.sub("", text)
    text = re.sub(
        r"^(?:\s*[\*\-]\s+\[.*?\]\(.*?\)\s*\n|\s*Tìm kiếm\.\.\.\s*\n|\s*__\s*\n|\s*\n)+(?=#)",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text.strip()


def purge(md_text: str) -> str:
    """
    Purge unwanted sections, replace images with [image file],
    and strip known Joomla boilerplate footers and top navigation.
    """
    text = clean_markdown(md_text)
    text = REGEX_REMOVE_MARKDOWN_IMAGE.sub("[image file]", text)
    text = text.replace(FOOTER_BOILERPLATE, "")
    text = strip_header_nav(text)
    return text


def create_html2text_converter(base_url: str = "") -> html2text.HTML2Text:
    """Configure an HTML2Text converter instance optimized for web scraping."""
    h = html2text.HTML2Text(baseurl=base_url)
    h.body_width = 0  # Disable text wrapping
    h.ignore_emphasis = False
    h.ignore_links = False
    h.ignore_images = False
    h.protect_links = False
    h.single_line_break = True
    h.mark_code = True
    h.escape_snob = False
    return h


def clean_html(html_content: str, base_url: str = "") -> str:
    """
    Convert raw HTML to Markdown and apply process_cleanup cleaning rules.

    1. Strips spam marquee tags and scripts/styles.
    2. Converts HTML to Markdown preserving headings, links, and tables.
    3. Purges unwanted sections (Sidebar, Đào tạo, E-Learning, Quy định, Kết nối).
    4. Replaces images with [image file].
    5. Strips footer boilerplate and top navigation bars.
    6. Normalizes line breaks.
    """
    if not html_content or not html_content.strip():
        return ""

    # Pre-strip hidden spam and scripts
    cleaned_html = REGEX_STRIP_MARQUEE.sub("", html_content)
    cleaned_html = REGEX_STRIP_SCRIPTS.sub("", cleaned_html)

    # Convert to markdown
    converter = create_html2text_converter(base_url=base_url)
    raw_md = converter.handle(cleaned_html)

    # Apply markdown purge rules
    cleaned_md = purge(raw_md)

    # Clean excessive empty lines
    cleaned_md = re.sub(r"\n{3,}", "\n\n", cleaned_md).strip()
    return cleaned_md


def clean_document_record(item: tuple[int, str, str]) -> tuple[int, str]:
    """Helper worker for thread pool: (doc_id, url, html) -> (doc_id, cleaned_markdown)."""
    doc_id, url, html_content = item
    try:
        md = clean_html(html_content, base_url=url)
        return doc_id, md
    except Exception as e:
        logger.warning(f"Error cleaning document {doc_id} ({url}): {e}")
        return doc_id, ""


def clean_database(
    db_path: str = DEFAULT_DB_PATH,
    batch_size: int = 500,
    max_workers: int = 8,
    only_missing: bool = False,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Process raw HTML in stash.db and save clean Markdown into the 'markdown' column.

    Args:
        db_path: Path to SQLite stash.db database.
        batch_size: Chunk size for reading and committing updates.
        max_workers: Number of threads for parallel conversion.
        only_missing: If True, only processes rows where markdown is NULL or empty.
        verbose: If True, display progress bar.

    Returns:
        Statistics dictionary with processed and updated counts.
    """
    resolved_path = str(Path(db_path).resolve())
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Database not found: {resolved_path}")

    conn = sqlite3.connect(resolved_path, timeout=60.0)
    # Enable WAL mode for high performance
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    cursor = conn.cursor()

    if only_missing:
        query_count = "SELECT count(*) FROM storedcrawlresult WHERE length(html) > 0 AND (markdown IS NULL OR length(markdown) = 0)"
        query_fetch = "SELECT id, url, html FROM storedcrawlresult WHERE length(html) > 0 AND (markdown IS NULL OR length(markdown) = 0)"
    else:
        query_count = "SELECT count(*) FROM storedcrawlresult WHERE length(html) > 0"
        query_fetch = "SELECT id, url, html FROM storedcrawlresult WHERE length(html) > 0"

    cursor.execute(query_count)
    total_docs = cursor.fetchone()[0]

    if verbose:
        mode_str = "missing rows only" if only_missing else "all rows"
        print(f"Cleaning raw HTML to Markdown in {db_path} ({total_docs} {mode_str}, {max_workers} workers)...")

    if total_docs == 0:
        if verbose:
            print("No matching rows to process.")
        conn.close()
        return {"total": 0, "updated": 0}

    cursor.execute(query_fetch)

    stats = {"total": total_docs, "updated": 0, "non_empty": 0}
    pbar = tqdm(total=total_docs, disable=not verbose, unit="doc", desc="Cleaning HTML")

    start_time = time.time()
    try:
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(clean_document_record, rows))

            # Batch update into sqlite
            update_data = [(md, doc_id) for doc_id, md in results]
            conn.executemany(
                "UPDATE storedcrawlresult SET markdown = ? WHERE id = ?",
                update_data,
            )
            conn.commit()

            stats["updated"] += len(results)
            stats["non_empty"] += sum(1 for _, md in results if md)
            pbar.update(len(results))

    finally:
        pbar.close()
        conn.close()

    elapsed = time.time() - start_time
    if verbose:
        print(
            f"Finished in {elapsed:.2f}s ({stats['updated']} rows updated, "
            f"{stats['non_empty']} non-empty markdowns)."
        )

    return stats


def strip_header_nav_from_database(
    db_path: str = DEFAULT_DB_PATH,
    batch_size: int = 500,
    verbose: bool = True,
) -> int:
    """
    Directly strip top header navigation menu from all records in the database.
    This runs in seconds without needing to re-parse the raw HTML.
    """
    resolved_path = str(Path(db_path).resolve())
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Database not found: {resolved_path}")

    conn = sqlite3.connect(resolved_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    cursor.execute("SELECT id, markdown FROM storedcrawlresult WHERE markdown LIKE '%* [FTU]%'")
    rows = cursor.fetchall()
    if verbose:
        print(f"Found {len(rows)} rows with header navigation to clean in {db_path}...")

    updates = []
    for doc_id, md in rows:
        cleaned = strip_header_nav(md)
        if cleaned != md:
            updates.append((cleaned, doc_id))

    for i in range(0, len(updates), batch_size):
        chunk = updates[i : i + batch_size]
        conn.executemany("UPDATE storedcrawlresult SET markdown = ? WHERE id = ?", chunk)
        conn.commit()

    conn.close()
    if verbose:
        print(f"Successfully cleaned {len(updates)} rows in {db_path}!")
    return len(updates)


def main() -> None:
    """CLI entry point for running HTML cleanup on stash.db."""
    parser = argparse.ArgumentParser(
        description="Clean raw HTML from stash.db and save clean Markdown into markdown column."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for SQLite transactions (default: 500)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only process rows where markdown is currently NULL or empty",
    )
    parser.add_argument(
        "--strip-nav-only",
        action="store_true",
        help="Quickly strip only the top navigation bar from markdown column",
    )

    args = parser.parse_args()

    if args.strip_nav_only:
        strip_header_nav_from_database(
            db_path=args.db,
            batch_size=args.batch_size,
            verbose=True,
        )
    else:
        clean_database(
            db_path=args.db,
            batch_size=args.batch_size,
            max_workers=args.workers,
            only_missing=args.only_missing,
            verbose=True,
        )


if __name__ == "__main__":
    main()
