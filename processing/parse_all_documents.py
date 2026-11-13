"""
Unified parser, OCR, and metadata extraction pipeline for stash.db.

Parses downloaded document assets:
- PDF (digital via pymupdf4llm, scanned via RapidOCR integration)
- Word DOCX (pandoc + html2text table normalization + XML metadata)
- Word DOC (antiword / catdoc + olefile OLE2 metadata)
- Excel XLSX / XLS / HTML (openpyxl / xlrd / html2text markdown tables + properties)
- Text documents

Updates `document_stubs` and `documents` tables:
- `documents.markdown = ...` (triggers FTS5 index sync)
- `documents.success = 1`
- `documents.metadata = {"title": ..., "mime_type": ..., "created_at": ..., "modified_at": ..., "author": ..., "producer": ...}`
- `document_stubs.status = 'extracted'`
- `document_stubs.extracted_chars = ...`
- `document_stubs.file_created_at`, `file_modified_at`, `file_author`, `file_producer`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure CUDA/cuDNN libraries from virtualenv are visible in LD_LIBRARY_PATH
_venv_root = Path(__file__).resolve().parent.parent / ".venv/lib"
_cudnn_paths = list(_venv_root.glob("*/site-packages/nvidia/cudnn/lib"))
if _cudnn_paths and _cudnn_paths[0].is_dir():
    _cudnn_dir = str(_cudnn_paths[0])
    _cublas_dir = str(_cudnn_paths[0].parent.parent / "cublas/lib")
    _current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _cudnn_dir not in _current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{_cudnn_dir}:{_cublas_dir}:{_current_ld}".rstrip(":")
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import concurrent.futures
import datetime
import json
import logging
import re
import sqlite3
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import html2text
import olefile
import openpyxl
import pymupdf
import pymupdf4llm
import xlrd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("parse_all_documents")

# Suppress MuPDF C-level warnings
try:
    pymupdf.TOOLS.mupdf_display_errors(False)
except Exception:
    pass

MIN_CHARS_PER_PAGE = 30
MIN_TOTAL_CHARS = 80


def sanitize_text(text: str) -> str:
    """Sanitize unpaired surrogates and null bytes for SQLite UTF-8 storage."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    return text.encode("utf-8", errors="replace").decode("utf-8")


def parse_iso_or_pdf_date(d: Any) -> str | None:
    """Normalize various date formats (PDF string, ISO string, datetime) to ISO 8601."""
    if not d:
        return None
    if isinstance(d, (datetime.datetime, datetime.date)):
        return d.isoformat()
    if isinstance(d, bytes):
        d = d.decode("utf-8", errors="ignore")
    d = str(d).strip()
    if not d:
        return None
    if d.startswith("D:"):
        d = d[2:]
    # Match YYYYMMDDHHmmss or YYYYMMDD
    m = re.match(r"^(\d{4})(\d{2})(\d{2})(?:(\d{2})(\d{2})(\d{2}))?", d)
    if m:
        year, month, day, hour, minute, second = m.groups()
        if hour is not None:
            return f"{year}-{month}-{day}T{hour}:{minute}:{second}"
        return f"{year}-{month}-{day}"
    # Standard ISO format clean-up
    m_iso = re.match(r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}:\d{2}))?", d)
    if m_iso:
        date_part, time_part = m_iso.groups()
        return f"{date_part}T{time_part}" if time_part else date_part
    return d[:50]


def clean_metadata_str(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="ignore")
    val = str(val).strip()
    return val if val else None


# =============================================================================
# Parsers for each file type
# =============================================================================

def is_pdf_digital(path: Path) -> bool:
    """Quick check if PDF has a selectable text layer."""
    try:
        doc = pymupdf.open(str(path))
        if len(doc) == 0:
            doc.close()
            return False
        total_raw_chars = sum(len(p.get_text().strip()) for p in doc)
        doc.close()
        return total_raw_chars >= MIN_TOTAL_CHARS
    except Exception:
        return False


def parse_pdf(path: Path, skip_ocr: bool = False) -> tuple[str, dict[str, Any]]:
    """Parse PDF using pymupdf4llm, falling back to RapidOCR if scanned."""
    meta: dict[str, Any] = {
        "created_at": None,
        "modified_at": None,
        "author": None,
        "producer": None,
    }
    try:
        doc = pymupdf.open(str(path))
    except Exception as e:
        return "", meta

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        return "", meta

    raw_meta = doc.metadata or {}
    meta["created_at"] = parse_iso_or_pdf_date(raw_meta.get("creationDate"))
    meta["modified_at"] = parse_iso_or_pdf_date(raw_meta.get("modDate"))
    meta["author"] = clean_metadata_str(raw_meta.get("author"))
    meta["producer"] = clean_metadata_str(raw_meta.get("producer") or raw_meta.get("creator"))

    total_raw_chars = sum(len(p.get_text().strip()) for p in doc)
    is_digital = total_raw_chars >= MIN_TOTAL_CHARS

    if skip_ocr and not is_digital:
        doc.close()
        return "", meta

    # Digital document with selectable text layer
    if is_digital:
        if page_count > 25:
            # For large digital PDFs (50-600 pages), fast text extraction avoids pymupdf4llm timeout
            pages_text = []
            for p in doc:
                t = p.get_text()
                if t.strip():
                    pages_text.append(t.strip())
            doc.close()
            md_text = "\n\n".join(pages_text)
            return sanitize_text(md_text).strip(), meta
        else:
            try:
                md_text = pymupdf4llm.to_markdown(doc, page_chunks=False, write_images=False)
                doc.close()
                return sanitize_text(md_text).strip(), meta
            except Exception as e:
                logger.warning("pymupdf4llm error on %s: %s; falling back to get_text", path, e)
                try:
                    doc2 = pymupdf.open(str(path))
                    plain = "\n\n".join(p.get_text() for p in doc2)
                    doc2.close()
                    return sanitize_text(plain).strip(), meta
                except Exception:
                    return "", meta

    # Scanned document (requires OCR)
    try:
        # pymupdf4llm uses RapidOCR (with CUDA enabled in backend)
        md_text = pymupdf4llm.to_markdown(doc, page_chunks=False, write_images=False)
        doc.close()
        return sanitize_text(md_text).strip(), meta
    except Exception as e:
        logger.warning("pymupdf4llm OCR error on %s: %s; falling back to plain get_text", path, e)
        try:
            doc2 = pymupdf.open(str(path))
            plain = "\n\n".join(p.get_text() for p in doc2)
            doc2.close()
            return sanitize_text(plain).strip(), meta
        except Exception:
            return "", meta


def parse_docx(path: Path) -> tuple[str, dict[str, Any]]:
    """Convert DOCX to clean Markdown via pandoc + html2text table normalization."""
    meta: dict[str, Any] = {
        "created_at": None,
        "modified_at": None,
        "author": None,
        "producer": None,
    }
    # 1. Extract metadata from core.xml
    try:
        with zipfile.ZipFile(path) as z:
            if "docProps/core.xml" in z.namelist():
                tree = ET.fromstring(z.read("docProps/core.xml"))
                ns = {
                    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
                    "dc": "http://purl.org/dc/elements/1.1/",
                    "dcterms": "http://purl.org/dc/terms/",
                }
                creator = tree.findtext("dc:creator", namespaces=ns)
                last_mod_by = tree.findtext("cp:lastModifiedBy", namespaces=ns)
                created = tree.findtext("dcterms:created", namespaces=ns)
                modified = tree.findtext("dcterms:modified", namespaces=ns)

                meta["author"] = clean_metadata_str(creator or last_mod_by)
                meta["created_at"] = parse_iso_or_pdf_date(created)
                meta["modified_at"] = parse_iso_or_pdf_date(modified)
                meta["producer"] = clean_metadata_str(last_mod_by)
    except Exception:
        pass

    # 2. Run pandoc
    try:
        res = subprocess.run(
            ["pandoc", "-f", "docx", "-t", "gfm", "--wrap=none", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        raw_md = res.stdout
    except Exception as e:
        logger.warning("pandoc error on %s: %s", path, e)
        return "", meta

    # 3. Clean and convert tables using html2text
    try:
        h = html2text.HTML2Text()
        h.body_width = 0
        h.ignore_images = True
        h.ignore_links = False
        h.ignore_emphasis = False
        clean_md = h.handle(raw_md)
        return sanitize_text(clean_md).strip(), meta
    except Exception:
        return sanitize_text(raw_md).strip(), meta


def parse_doc(path: Path) -> tuple[str, dict[str, Any]]:
    """Extract legacy Word .doc using antiword, catdoc, and olefile."""
    meta: dict[str, Any] = {
        "created_at": None,
        "modified_at": None,
        "author": None,
        "producer": None,
    }
    # 1. Metadata via olefile
    try:
        if olefile.isOleFile(path):
            ole = olefile.OleFileIO(path)
            m = ole.get_metadata()
            meta["author"] = clean_metadata_str(m.author or m.last_saved_by)
            meta["created_at"] = parse_iso_or_pdf_date(m.create_time)
            meta["modified_at"] = parse_iso_or_pdf_date(m.last_saved_time)
            meta["producer"] = clean_metadata_str(m.creating_application)
            ole.close()
    except Exception:
        pass

    # 2. Try antiword (preserves layout and tables)
    try:
        res = subprocess.run(
            ["antiword", "-w", "0", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if res.returncode == 0 and len(res.stdout.strip()) > 30:
            return sanitize_text(res.stdout).strip(), meta
    except Exception:
        pass

    # 3. Fallback to catdoc
    try:
        res = subprocess.run(
            ["catdoc", "-dutf-8", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if res.returncode == 0 and len(res.stdout.strip()) > 10:
            return sanitize_text(res.stdout).strip(), meta
    except Exception:
        pass

    return "", meta


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format headers and rows into a clean Markdown table."""
    if not headers and not rows:
        return ""
    if not headers and rows:
        headers = rows[0]
        rows = rows[1:]

    # Clean cell values
    def clean_cell(c: str) -> str:
        c = str(c).replace("\n", " ").replace("\r", " ").replace("|", "\\|").strip()
        return c

    clean_headers = [clean_cell(h) for h in headers]
    # Ensure at least 1 column
    if not clean_headers:
        return ""

    num_cols = len(clean_headers)
    lines = [
        "| " + " | ".join(clean_headers) + " |",
        "| " + " | ".join(["---"] * num_cols) + " |",
    ]

    for row in rows:
        padded = [clean_cell(row[i]) if i < len(row) else "" for i in range(num_cols)]
        if any(padded):  # Skip completely empty rows
            lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(lines)


def parse_xlsx(path: Path) -> tuple[str, dict[str, Any]]:
    """Convert XLSX to Markdown tables sheet-by-sheet."""
    meta: dict[str, Any] = {
        "created_at": None,
        "modified_at": None,
        "author": None,
        "producer": None,
    }
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
        p = wb.properties
        if p:
            meta["author"] = clean_metadata_str(p.creator or p.lastModifiedBy)
            meta["created_at"] = parse_iso_or_pdf_date(p.created)
            meta["modified_at"] = parse_iso_or_pdf_date(p.modified)
            meta["producer"] = clean_metadata_str(p.lastModifiedBy)
    except Exception:
        pass

    try:
        wb = openpyxl.load_workbook(path, data_only=True)
    except Exception as e:
        logger.warning("openpyxl error on %s: %s", path, e)
        return "", meta

    sheet_sections = []
    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        raw_rows = list(sheet.iter_rows(values_only=True))
        if not raw_rows:
            continue

        # Filter out empty rows
        clean_rows = []
        for r in raw_rows:
            str_row = [str(c).strip() if c is not None else "" for c in r]
            if any(str_row):
                clean_rows.append(str_row)

        if not clean_rows:
            continue

        # Trim empty trailing columns
        max_col = 0
        for r in clean_rows:
            for idx in range(len(r) - 1, -1, -1):
                if r[idx]:
                    max_col = max(max_col, idx + 1)
                    break
        if max_col == 0:
            continue

        trimmed_rows = [r[:max_col] for r in clean_rows]
        headers = trimmed_rows[0]
        data_rows = trimmed_rows[1:]

        table_md = format_markdown_table(headers, data_rows)
        if table_md:
            sheet_sections.append(f"## Bảng: {sheet_name}\n\n{table_md}")

    content = "\n\n".join(sheet_sections)
    return sanitize_text(content).strip(), meta


def parse_xls(path: Path) -> tuple[str, dict[str, Any]]:
    """Convert legacy XLS to Markdown tables."""
    meta: dict[str, Any] = {
        "created_at": None,
        "modified_at": None,
        "author": None,
        "producer": None,
    }
    # Metadata via olefile
    try:
        if olefile.isOleFile(path):
            ole = olefile.OleFileIO(path)
            m = ole.get_metadata()
            meta["author"] = clean_metadata_str(m.author or m.last_saved_by)
            meta["created_at"] = parse_iso_or_pdf_date(m.create_time)
            meta["modified_at"] = parse_iso_or_pdf_date(m.last_saved_time)
            meta["producer"] = clean_metadata_str(m.creating_application)
            ole.close()
    except Exception:
        pass

    try:
        wb = xlrd.open_workbook(str(path))
    except Exception as e:
        logger.warning("xlrd error on %s: %s", path, e)
        return "", meta

    sheet_sections = []
    for sheet_name in wb.sheet_names():
        sh = wb.sheet_by_name(sheet_name)
        if sh.nrows == 0 or sh.ncols == 0:
            continue

        clean_rows = []
        for r_idx in range(sh.nrows):
            row_vals = []
            for c_idx in range(sh.ncols):
                val = sh.cell_value(r_idx, c_idx)
                if val is None or val == "":
                    row_vals.append("")
                elif isinstance(val, float) and val.is_integer():
                    row_vals.append(str(int(val)))
                else:
                    row_vals.append(str(val).strip())
            if any(row_vals):
                clean_rows.append(row_vals)

        if not clean_rows:
            continue

        # Trim trailing empty columns
        max_col = 0
        for r in clean_rows:
            for idx in range(len(r) - 1, -1, -1):
                if r[idx]:
                    max_col = max(max_col, idx + 1)
                    break
        if max_col == 0:
            continue

        trimmed_rows = [r[:max_col] for r in clean_rows]
        headers = trimmed_rows[0]
        data_rows = trimmed_rows[1:]

        table_md = format_markdown_table(headers, data_rows)
        if table_md:
            sheet_sections.append(f"## Bảng: {sheet_name}\n\n{table_md}")

    content = "\n\n".join(sheet_sections)
    return sanitize_text(content).strip(), meta


def parse_html_table(path: Path) -> tuple[str, dict[str, Any]]:
    """Parse pubhtml Google Sheets or forms using html2text."""
    meta: dict[str, Any] = {
        "created_at": None,
        "modified_at": None,
        "author": None,
        "producer": None,
    }
    raw_html = path.read_text(encoding="utf-8", errors="ignore")
    h = html2text.HTML2Text()
    h.body_width = 0
    h.ignore_images = True
    h.ignore_links = False
    md = h.handle(raw_html)
    return sanitize_text(md).strip(), meta


def parse_text_file(path: Path) -> tuple[str, dict[str, Any]]:
    """Parse text file."""
    meta: dict[str, Any] = {
        "created_at": None,
        "modified_at": None,
        "author": None,
        "producer": None,
    }
    content = path.read_text(encoding="utf-8", errors="ignore")
    return sanitize_text(content).strip(), meta


def dispatch_parser(file_type: str, path: Path, skip_ocr: bool = False) -> tuple[str, dict[str, Any]]:
    """Dispatch file to the appropriate parser based on extension or mime type."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, skip_ocr=skip_ocr)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix == ".doc":
        return parse_doc(path)
    if suffix == ".xlsx":
        return parse_xlsx(path)
    if suffix == ".xls":
        return parse_xls(path)
    if suffix in (".html", ".htm"):
        return parse_html_table(path)
    if suffix in (".txt", ".md"):
        return parse_text_file(path)

    # Fallback to suffix-less detection
    if "pdf" in file_type:
        return parse_pdf(path, skip_ocr=skip_ocr)
    if "document" in file_type or "word" in file_type:
        return parse_docx(path)
    if "spreadsheet" in file_type or "sheet" in file_type or "excel" in file_type:
        return parse_xlsx(path)

    return "", {"created_at": None, "modified_at": None, "author": None, "producer": None}


# =============================================================================
# Database & Pipeline
# =============================================================================

def process_single_stub(row: tuple, skip_ocr: bool = False, scanned_only: bool = False) -> dict[str, Any]:
    doc_id, url, title, mime_type, local_path, cur_created, cur_modified, cur_author, cur_producer = row
    dest_path = Path(local_path)
    if not dest_path.exists():
        return {
            "doc_id": doc_id,
            "success": False,
            "error": "File not found on disk",
            "chars": 0,
            "markdown": "",
            "meta": {},
        }

    suffix = dest_path.suffix.lower()
    if suffix == ".pdf":
        is_digital = is_pdf_digital(dest_path)
        if skip_ocr and not is_digital:
            return {"doc_id": doc_id, "skipped": True, "success": False, "error": "Skipped (scanned PDF)"}
        if scanned_only and is_digital:
            return {"doc_id": doc_id, "skipped": True, "success": False, "error": "Skipped (digital PDF)"}

    try:
        md_text, meta = dispatch_parser(mime_type, dest_path, skip_ocr=skip_ocr)
    except Exception as e:
        return {
            "doc_id": doc_id,
            "success": False,
            "error": str(e),
            "chars": 0,
            "markdown": "",
            "meta": {},
        }

    # Merge metadata with existing values
    final_meta = {
        "title": title,
        "mime_type": mime_type,
        "created_at": meta.get("created_at") or cur_created,
        "modified_at": meta.get("modified_at") or cur_modified,
        "author": meta.get("author") or cur_author,
        "producer": meta.get("producer") or cur_producer,
    }

    return {
        "doc_id": doc_id,
        "title": title,
        "success": len(md_text.strip()) > 0,
        "error": None if len(md_text.strip()) > 0 else "Extracted 0 characters",
        "chars": len(md_text),
        "markdown": md_text,
        "meta": final_meta,
    }


def run_pipeline(
    db_path: str = "stash.db",
    file_type: str = "all",
    workers: int = 8,
    limit: int | None = None,
    doc_id: int | None = None,
    force: bool = False,
    digital_only: bool = False,
    scanned_only: bool = False,
) -> None:
    conn = sqlite3.connect(db_path, timeout=60.0)
    cur = conn.cursor()

    type_filter = ""
    if file_type == "pdf":
        type_filter = "AND ds.type = 'application/pdf'"
    elif file_type == "docx":
        type_filter = "AND (ds.type LIKE '%wordprocessingml%' OR ds.type = 'application/vnd.google-apps.document')"
    elif file_type == "doc":
        type_filter = "AND ds.type = 'application/msword'"
    elif file_type == "excel":
        type_filter = "AND (ds.type LIKE '%spreadsheet%' OR ds.type LIKE '%excel%')"
    elif file_type == "word":
        type_filter = "AND (ds.type LIKE '%word%' OR ds.type LIKE '%document%') AND ds.type NOT LIKE '%spreadsheet%'"

    status_filter = "" if force else "AND ds.status != 'extracted'"

    if doc_id:
        query = f"""
            SELECT ds.doc_id, ds.url, ds.title, ds.type, ds.local_path,
                   ds.file_created_at, ds.file_modified_at, ds.file_author, ds.file_producer
            FROM document_stubs ds
            WHERE ds.doc_id = ?
        """
        cur.execute(query, (doc_id,))
    else:
        query = f"""
            SELECT ds.doc_id, ds.url, ds.title, ds.type, ds.local_path,
                   ds.file_created_at, ds.file_modified_at, ds.file_author, ds.file_producer
            FROM document_stubs ds
            WHERE ds.local_path IS NOT NULL AND length(ds.local_path) > 0
              {status_filter}
              {type_filter}
            ORDER BY ds.doc_id
        """
        cur.execute(query)

    rows = cur.fetchall()
    if limit:
        rows = rows[:limit]

    total = len(rows)
    logger.info("Found %d document stubs to parse (type=%s, workers=%d, digital_only=%s, scanned_only=%s)...", total, file_type, workers, digital_only, scanned_only)
    if total == 0:
        logger.info("No documents need parsing.")
        conn.close()
        return

    # Track statistics
    stats = {"success": 0, "failed": 0, "skipped": 0, "total_chars": 0}
    t0 = time.time()
    batch_size = 25
    batch_updates = []

    def commit_batch(updates: list[dict]):
        nonlocal conn
        cur = conn.cursor()
        for res in updates:
            d_id = res["doc_id"]
            if res.get("skipped"):
                continue
            if res["success"]:
                md = res["markdown"]
                m = res["meta"]
                meta_json = json.dumps(m, ensure_ascii=False)
                # Update documents table (triggers FTS update)
                cur.execute(
                    "UPDATE documents SET markdown = ?, success = 1, metadata = ? WHERE id = ?",
                    (md, meta_json, d_id),
                )
                # Update document_stubs table
                cur.execute(
                    """
                    UPDATE document_stubs
                    SET status = 'extracted',
                        extracted_chars = ?,
                        file_created_at = COALESCE(?, file_created_at),
                        file_modified_at = COALESCE(?, file_modified_at),
                        file_author = COALESCE(?, file_author),
                        file_producer = COALESCE(?, file_producer)
                    WHERE doc_id = ?
                    """,
                    (res["chars"], m.get("created_at"), m.get("modified_at"), m.get("author"), m.get("producer"), d_id),
                )
            else:
                cur.execute(
                    "UPDATE document_stubs SET status = 'failed' WHERE doc_id = ?",
                    (d_id,),
                )
        conn.commit()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_stub, r, digital_only, scanned_only): r for r in rows}
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            completed += 1
            if res.get("skipped"):
                stats["skipped"] += 1
            elif res["success"]:
                stats["success"] += 1
                stats["total_chars"] += res["chars"]
            else:
                stats["failed"] += 1
                logger.warning("Doc #%d (%s) failed: %s", res["doc_id"], res.get("title", ""), res["error"])

            if not res.get("skipped"):
                batch_updates.append(res)
                if len(batch_updates) >= batch_size:
                    commit_batch(batch_updates)
                    batch_updates.clear()

            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    "[%d/%d] (%.1f%%) Success: %d, Skipped: %d, Failed: %d, Total Chars: %s (%.1f docs/s)",
                    completed,
                    total,
                    (completed / total) * 100,
                    stats["success"],
                    stats["skipped"],
                    stats["failed"],
                    f"{stats['total_chars']:,}",
                    rate,
                )

    if batch_updates:
        commit_batch(batch_updates)
        batch_updates.clear()

    conn.close()
    elapsed = time.time() - t0
    logger.info("Done! Processed %d documents in %.1fs. Success: %d, Skipped: %d, Failed: %d", total, elapsed, stats["success"], stats["skipped"], stats["failed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse all documents into Markdown and update stash.db")
    parser.add_argument("--type", choices=["all", "pdf", "docx", "doc", "excel", "word"], default="all", help="File type filter")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--doc-id", type=int, default=None, help="Specific doc_id to test")
    parser.add_argument("--force", action="store_true", help="Force re-parsing of already extracted files")
    parser.add_argument("--digital-only", action="store_true", help="Only process digital documents (skip OCR)")
    parser.add_argument("--scanned-only", action="store_true", help="Only process scanned PDFs (require OCR)")
    parser.add_argument("--db-path", default="stash.db", help="Path to stash.db")

    args = parser.parse_args()
    run_pipeline(
        db_path=args.db_path,
        file_type=args.type,
        workers=args.workers,
        limit=args.limit,
        doc_id=args.doc_id,
        force=args.force,
        digital_only=args.digital_only,
        scanned_only=args.scanned_only,
    )
