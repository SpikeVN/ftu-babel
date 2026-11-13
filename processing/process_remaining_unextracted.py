"""
process_remaining_unextracted.py - OCR and extract all remaining valid document assets.

1. OCR Doc #9527 (scanned 6-page PDF).
2. Extract Doc #9105, #9106, #9112, #9113 (Word DOCX templates with textboxes).
3. Unpack and extract Doc #10254, #10259, #10295, #10301 (ZIP archives with inner DOC/DOCX documents).
4. Update document_stubs, documents, document_chunks, vec_chunks, and fts_documents.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure CUDA/cuDNN visible
_venv_root = Path(".venv/lib").resolve()
_cudnn_paths = list(_venv_root.glob("*/site-packages/nvidia/cudnn/lib"))
if _cudnn_paths and _cudnn_paths[0].is_dir():
    _cudnn_dir = str(_cudnn_paths[0])
    _cublas_dir = str(_cudnn_paths[0].parent.parent / "cublas/lib")
    _current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _cudnn_dir not in _current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{_cudnn_dir}:{_cublas_dir}:{_current_ld}".rstrip(":")
        os.execv(sys.executable, [sys.executable] + sys.argv)

import fitz
import sqlite_vec
from rapidocr_onnxruntime import RapidOCR
from processing.embeddings import run_hierarchical_chunk_embedding
from processing.parse_all_documents import parse_doc, parse_docx, sanitize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("process_remaining")

DB_PATH = "stash.db"


def open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def ocr_pdf_file(pdf_path: Path) -> str:
    """Run RapidOCR on all pages of a scanned PDF."""
    logger.info(f"Running RapidOCR on {pdf_path}...")
    ocr = RapidOCR()
    doc = fitz.open(str(pdf_path))
    pages_text = []

    for pno, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        result, elapse = ocr(img_bytes)
        ptext = ""
        if result:
            ptext = "\n".join([line[1] for line in result])
        logger.info(f"  Page {pno+1}/{len(doc)}: {len(ptext)} chars in {sum(elapse):.2f}s")
        pages_text.append(f"### Trang {pno+1}\n\n{ptext}")

    doc.close()
    return "\n\n".join(pages_text).strip()


def extract_docx_with_textboxes(docx_path: Path) -> str:
    """Extract text from docx including textboxes and drawings."""
    with zipfile.ZipFile(docx_path) as z:
        if "word/document.xml" not in z.namelist():
            return ""
        xml_content = z.read("word/document.xml")

    # Strip XML tags while preserving whitespace
    raw_text = re.sub(r"<[^>]+>", " ", xml_content.decode("utf-8", errors="ignore"))
    clean_text = re.sub(r"\s+", " ", raw_text).strip()
    return clean_text


def extract_zip_archive(zip_path: Path) -> str:
    """Unpack inner doc/docx files from a zip archive and extract all text."""
    logger.info(f"Extracting inner documents from {zip_path}...")
    extracted_sections = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmpdir)

        for f in sorted(tmp_path.rglob("*")):
            if f.name.startswith("._") or f.is_dir():
                continue
            ext = f.suffix.lower()
            rel_name = f.relative_to(tmp_path).as_posix()

            text = ""
            if ext == ".docx":
                t, _ = parse_docx(f)
                if not t.strip():
                    t = extract_docx_with_textboxes(f)
                text = t.strip()
            elif ext == ".doc":
                t, _ = parse_doc(f)
                text = t.strip()

            if text:
                logger.info(f"  - Extracted {len(text)} chars from inner file: {rel_name}")
                extracted_sections.append(f"## Tệp đính kèm: {rel_name}\n\n{text}")

    return "\n\n---\n\n".join(extracted_sections).strip()


def main():
    conn = open_db()
    cursor = conn.cursor()
    processed_doc_ids = []

    # 1. OCR Doc #9527
    pdf_path = Path("downloads/pdfs/9527.pdf")
    if pdf_path.exists():
        ocr_text = ocr_pdf_file(pdf_path)
        if len(ocr_text) > 100:
            doc_id = 9527
            cursor.execute(
                "UPDATE document_stubs SET status = 'extracted', extracted_chars = ? WHERE doc_id = ?",
                (len(ocr_text), doc_id),
            )
            title = "Nghị quyết số 248/2025/QH15 về cơ chế, chính sách đặc thù phát triển GD&ĐT"
            cursor.execute(
                "UPDATE documents SET markdown = ?, success = 1, metadata = json_set(metadata, '$.title', ?) WHERE id = ?",
                (ocr_text, title, doc_id),
            )
            processed_doc_ids.append(doc_id)
            logger.info(f"Updated Doc #{doc_id} with {len(ocr_text)} OCR chars")

    # 2. Extract Doc #9105, #9106, #9112, #9113
    docx_ids = [9105, 9106, 9112, 9113]
    for did in docx_ids:
        p = Path(f"downloads/docx/{did}.docx")
        if p.exists():
            text = extract_docx_with_textboxes(p)
            if len(text) > 100:
                cursor.execute(
                    "UPDATE document_stubs SET status = 'extracted', extracted_chars = ? WHERE doc_id = ?",
                    (len(text), did),
                )
                cursor.execute(
                    "UPDATE documents SET markdown = ?, success = 1 WHERE id = ?",
                    (text, did),
                )
                processed_doc_ids.append(did)
                logger.info(f"Updated Doc #{did} with {len(text)} chars")

    # 3. Extract Doc #10254, #10259, #10295, #10301
    zip_ids = [10254, 10259, 10295, 10301]
    for did in zip_ids:
        p = Path(f"downloads/archives/{did}.zip")
        if p.exists():
            text = extract_zip_archive(p)
            if len(text) > 100:
                cursor.execute(
                    "UPDATE document_stubs SET status = 'extracted', extracted_chars = ? WHERE doc_id = ?",
                    (len(text), did),
                )
                cursor.execute(
                    "UPDATE documents SET markdown = ?, success = 1 WHERE id = ?",
                    (text, did),
                )
                processed_doc_ids.append(did)
                logger.info(f"Updated Doc #{did} with {len(text)} chars from archive")

    conn.commit()

    # 4. Sync FTS5 for all processed documents
    logger.info(f"Syncing FTS5 for {len(processed_doc_ids)} newly extracted documents...")
    for did in processed_doc_ids:
        cursor.execute("SELECT url, json_extract(metadata, '$.title'), markdown FROM documents WHERE id = ?", (did,))
        r = cursor.fetchone()
        if r:
            url, title, md = r
            cursor.execute("DELETE FROM fts_documents WHERE doc_id = ?", (did,))
            cursor.execute(
                "INSERT INTO fts_documents (doc_id, url, title, content, summary) VALUES (?, ?, ?, ?, ?)",
                (did, url, title or "", md or "", ""),
            )
    conn.commit()
    conn.close()

    # 5. Re-embed newly extracted documents
    if processed_doc_ids:
        logger.info(f"Re-embedding {len(processed_doc_ids)} documents into vec_chunks...")
        run_hierarchical_chunk_embedding(
            db_path=DB_PATH,
            skip_existing=True,
            batch_size=64,
            concurrency=8,
        )

    logger.info(f"=== Complete! Successfully processed and embedded {len(processed_doc_ids)} documents ===")


if __name__ == "__main__":
    main()
