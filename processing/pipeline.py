"""
Unified Periodic Processing Pipeline for Babel Knowledge Base.

Streamlines and orchestrates:
1. Link Canonicalization & External Document Stubbing (internal links -> /docs/{id})
2. Document Asset Fetching (PDF, DOCX, XLSX, Google Drive / Docs / Sheets, Dropbox)
3. Document Parsing & GPU OCR (PyMuPDF, RapidOCR GPU, Pandoc, Antiword, OpenPyXL, ZIP)
4. Image Template Filtering & Diagram VLM Transcription (Mermaid + Indented + Prose)
5. Hierarchical Small-to-Big Chunking & Vector Embedding (sqlite-vec + FTS5 sync)
6. Unscraped Candidate Link Export (CSV extraction for future crawler runs)

All stages are fully idempotent (skip-existing) so this script can be executed
periodically without duplicating work or re-embedding existing documents.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup dynamic linker path for GPU RapidOCR if cudnn is present
def ensure_gpu_environment():
    site_packages = PROJECT_ROOT / ".venv" / "lib" / "python3.14" / "site-packages"
    cudnn_lib = site_packages / "nvidia" / "cudnn" / "lib"
    cublas_lib = site_packages / "nvidia" / "cublas" / "lib"
    extra_paths = []
    if cudnn_lib.exists():
        extra_paths.append(str(cudnn_lib))
    if cublas_lib.exists():
        extra_paths.append(str(cublas_lib))
    if extra_paths:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(extra_paths + ([current_ld] if current_ld else []))
        os.environ["LD_LIBRARY_PATH"] = new_ld

ensure_gpu_environment()

import sqlite_vec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")

try:
    from server.config import (
        DEFAULT_DB_PATH,
        DEFAULT_EMBED_API_KEY as DEFAULT_EMBED_KEY,
        DEFAULT_EMBED_MODEL,
        DEFAULT_EMBED_URL,
        DEFAULT_USEARCH_PATH as DEFAULT_INDEX_PATH,
        MOSS_API_KEY,
        MOSS_EMBEDDING_ENDPOINT,
        MOSS_RERANKING_ENDPOINT,
    )
except ImportError:
    DEFAULT_DB_PATH = str(PROJECT_ROOT / "stash.db")
    DEFAULT_INDEX_PATH = str(PROJECT_ROOT / "stash.usearch")
    MOSS_API_KEY = os.environ.get("MOSS_API_KEY") or os.environ.get("FTULABS_API_KEY", "")
    MOSS_EMBEDDING_ENDPOINT = (
        os.environ.get("MOSS_EMBEDDING_ENDPOINT")
        or os.environ.get("FTULABS_API_ENDPOINT", "https://550w.falcolabs.org/v1/")
    ).rstrip("/")
    MOSS_RERANKING_ENDPOINT = (
        os.environ.get("MOSS_RERANKING_ENDPOINT")
        or os.environ.get("FTULABS_API_ENDPOINT", "https://550w.falcolabs.org/v1/")
    ).rstrip("/")
    DEFAULT_EMBED_URL = os.environ.get(
        "EMBED_URL",
        f"{MOSS_EMBEDDING_ENDPOINT}/embeddings" if not MOSS_EMBEDDING_ENDPOINT.endswith("/v1") else f"{MOSS_EMBEDDING_ENDPOINT}/embeddings",
    )
    DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "unsloth/embeddinggemma-2-GGUF")
    DEFAULT_EMBED_KEY = os.environ.get("EMBED_API_KEY") or MOSS_API_KEY

DEFAULT_VLM_URL = os.environ.get("VLM_URL", "https://swarm.ftds.online/api/chat/completions")
DEFAULT_VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen3.6-35B-A3B")

SOCIAL_DOMAINS = {
    "facebook.com", "m.facebook.com", "l.facebook.com", "fb.me", "fb.com",
    "youtube.com", "youtu.be", "instagram.com", "linkedin.com",
    "twitter.com", "x.com", "tiktok.com", "zalo.me", "threads.net",
    "pinterest.com",
}

DOC_EXT_REGEX = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|rar|zip)(\?.*)?$", re.IGNORECASE)
IMG_EXT_REGEX = re.compile(r"\.(jpg|jpeg|png|gif|webp|svg|ico|bmp)(\?.*)?$", re.IGNORECASE)
MD_LINK_REGEX = re.compile(r'\[([^\]]+)\]\((https?://[^\s\"\)]+)(?:\s+\"([^\"]*)\")?\)')
GENERIC_TITLES = {
    "", "tại đây", "tại đây.", "link", "click here", "click here!",
    "xem tại đây", "tải về", "xem chi tiết", "đây", "xem", "download",
}


def open_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open SQLite database with sqlite_vec extension loaded."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    return conn


def normalize_url(u: str) -> str:
    """Normalize URL for consistent lookup."""
    if not u:
        return ""
    u = unquote(u.strip()).replace(chr(92), "/")
    p = urlparse(u)
    scheme = "https" if p.scheme in ("http", "https") else p.scheme
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = p.path.rstrip("/")
    query = p.query
    if "google.com" in netloc and query:
        return f"{scheme}://{netloc}{path}?{query}"
    return f"{scheme}://{netloc}{path}"


def is_social(url_lower: str) -> bool:
    return any(domain in url_lower for domain in SOCIAL_DOMAINS)


# ==============================================================================
# STAGE 1: LINK CANONICALIZATION & STUBBING
# ==============================================================================
def run_link_canonicalization_and_stubs(db_path: str = DEFAULT_DB_PATH, dry_run: bool = False) -> dict[str, int]:
    """
    1. Rewrites markdown links pointing to existing documents to /docs/{doc_id}.
    2. Identifies external document links (PDFs, DOCX, XLSX, Google Drive/Docs/Sheets),
       creates clean stub records in documents & document_stubs, and rewrites them to /docs/{stub_id}.
    3. Preserves social media links as external URLs.
    """
    logger.info("=== STAGE 1: Link Canonicalization & Document Stubbing ===")
    conn = open_db(db_path)
    cur = conn.cursor()

    # Build URL map
    cur.execute("SELECT id, url, redirected_url FROM documents")
    doc_map: dict[str, int] = {}
    for did, u, ru in cur.fetchall():
        if u:
            doc_map[normalize_url(u)] = did
        if ru:
            doc_map[normalize_url(ru)] = did

    cur.execute("SELECT doc_id, url FROM document_stubs")
    for did, u in cur.fetchall():
        if u:
            norm = normalize_url(u)
            if norm not in doc_map:
                doc_map[norm] = did

    logger.info("Loaded %d known URLs from database.", len(doc_map))

    # Scan markdown for un-stubbed document assets
    cur.execute("SELECT id, markdown FROM documents WHERE markdown IS NOT NULL")
    all_docs = cur.fetchall()

    new_stubs_to_create: dict[str, dict[str, Any]] = {}
    for doc_id, md in all_docs:
        for m in MD_LINK_REGEX.finditer(md):
            text = m.group(1).strip()
            href = m.group(2).strip()
            href_l = href.lower()

            if is_social(href_l) or IMG_EXT_REGEX.search(href_l):
                continue

            norm = normalize_url(href)
            if norm in doc_map:
                continue

            is_doc_file = bool(DOC_EXT_REGEX.search(href_l))
            is_gdrive = "drive.google.com" in href_l or "docs.google.com" in href_l

            if is_doc_file or is_gdrive:
                ftype = "google_drive" if is_gdrive else "pdf_or_office"
                if norm not in new_stubs_to_create:
                    new_stubs_to_create[norm] = {
                        "url": href,
                        "title": text,
                        "parent_id": doc_id,
                        "type": ftype,
                    }

    new_stubs_count = 0
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    if not dry_run and new_stubs_to_create:
        logger.info("Creating %d new document stubs...", len(new_stubs_to_create))
        for norm, item in new_stubs_to_create.items():
            raw_url = item["url"]
            title = item["title"]
            if not title or title.lower() in GENERIC_TITLES:
                title = f"Tài liệu tham chiếu #{item['parent_id']}"
            parent_id = item["parent_id"]
            ftype = item["type"]

            stub_md = (
                f"# {title}\n\n"
                f"- **Nguồn tài liệu**: [{raw_url}]({raw_url})\n"
                f"- **Tài liệu tham chiếu từ bài viết**: [/docs/{parent_id}](/docs/{parent_id})\n"
                f"- **Loại tệp**: {ftype}\n\n"
                f"*Chờ trích xuất văn bản chi tiết và tạo vector embedding.*"
            )
            meta_json = json.dumps({
                "title": title,
                "source": ftype,
                "parent_doc_id": parent_id,
                "status": "pending_text_extraction",
            }, ensure_ascii=False)

            cur.execute(
                """
                INSERT INTO documents (
                    url, html, success, media, links, tables,
                    markdown, metadata, saved_at
                ) VALUES (?, '', 1, '{}', '{}', '[]', ?, ?, ?)
                """,
                (raw_url, stub_md, meta_json, now_str)
            )
            new_id = cur.lastrowid
            doc_map[norm] = new_id

            # Insert into document_stubs
            cur.execute(
                """
                INSERT OR IGNORE INTO document_stubs (
                    doc_id, url, title, file_type, source_domain, parent_doc_id, status, type
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (new_id, raw_url, title, ftype, urlparse(raw_url).netloc, parent_id, ftype)
            )
            new_stubs_count += 1

        conn.commit()

    # Rewrite markdown links across all documents
    total_rewrites = 0
    updated_docs = 0

    def link_replacer(doc_id: int):
        def _rep(match: re.Match) -> str:
            nonlocal total_rewrites
            text = match.group(1)
            href = match.group(2)
            title = match.group(3)
            href_l = href.lower()

            if is_social(href_l) or IMG_EXT_REGEX.search(href_l):
                return match.group(0)

            norm = normalize_url(href)
            if norm in doc_map and doc_id != doc_map[norm]:
                target_id = doc_map[norm]
                total_rewrites += 1
                t_part = f' "{title}"' if title else ""
                return f"[{text}](/docs/{target_id}{t_part})"

            return match.group(0)
        return _rep

    for doc_id, md in all_docs:
        new_md = MD_LINK_REGEX.sub(link_replacer(doc_id), md)
        if new_md != md:
            updated_docs += 1
            if not dry_run:
                cur.execute("UPDATE documents SET markdown = ? WHERE id = ?", (new_md, doc_id))

    if not dry_run:
        conn.commit()

    conn.close()
    logger.info("Stage 1 complete: %d new stubs created, %d links rewritten across %d docs.",
                new_stubs_count, total_rewrites, updated_docs)
    return {"new_stubs": new_stubs_count, "rewritten_links": total_rewrites, "updated_docs": updated_docs}


# ==============================================================================
# STAGE 2: FETCH PENDING DOCUMENT ASSETS
# ==============================================================================
def run_fetch_stubs(db_path: str = DEFAULT_DB_PATH, force: bool = False) -> dict[str, int]:
    """
    Downloads document stubs that have status = 'pending'.
    Uses proper endpoints for Google Drive direct downloads and Google Docs/Sheets export.
    """
    logger.info("=== STAGE 2: Fetching Pending Document Stubs ===")
    from processing.fetch_pdfs import run_fetch as run_fetch_pdfs
    from processing.fetch_other_files import run_pipeline as run_fetch_other

    conn = open_db(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM document_stubs WHERE status = 'pending'")
    pending_count = cur.fetchone()[0]
    conn.close()

    logger.info("Total pending stubs to fetch: %d", pending_count)
    if pending_count == 0 and not force:
        logger.info("No pending stubs to fetch. Skipping.")
        return {"fetched": 0}

    # 1. Fetch PDFs
    asyncio.run(run_fetch_pdfs(db_path=db_path, force=force))
    # 2. Fetch Other Files (DOCX, Spreadsheets, Google Drive/Docs)
    asyncio.run(run_fetch_other(db_path=db_path, force=force))

    return {"pending_processed": pending_count}


# ==============================================================================
# STAGE 3: DOCUMENT PARSING & GPU OCR
# ==============================================================================
def run_parse_documents(
    db_path: str = DEFAULT_DB_PATH,
    concurrency: int = 4,
    skip_ocr: bool = False,
    force: bool = False,
) -> dict[str, int]:
    """
    Parses downloaded documents (PDF, DOCX, DOC, XLSX, XLS, TXT, ZIP).
    Accelerated with RapidOCR GPU for scanned pages.
    """
    logger.info("=== STAGE 3: Document Parsing & GPU OCR ===")
    from processing.parse_all_documents import run_pipeline as run_parse_pipeline

    conn = open_db(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) FROM document_stubs
        WHERE (status = 'downloaded' OR (status = 'extracted' AND extracted_chars = 0))
          AND local_path IS NOT NULL
        """
    )
    to_parse = cur.fetchone()[0]
    conn.close()

    logger.info("Documents requiring parsing/OCR: %d", to_parse)
    if to_parse == 0 and not force:
        logger.info("All downloaded stubs are already parsed. Skipping.")
        return {"parsed": 0}

    result = run_parse_pipeline(
        db_path=db_path,
        workers=concurrency,
        force=force,
    )
    return {"parsed": to_parse}


# ==============================================================================
# STAGE 4: IMAGE TEMPLATE FILTERING & DIAGRAM VLM TRANSCRIPTION
# ==============================================================================
def run_image_diagram_pipeline(db_path: str = DEFAULT_DB_PATH, dry_run: bool = False) -> dict[str, int]:
    """
    1. Cleans up repetitive template markers (flags, logos, UI banners) across articles.
    2. Identifies candidate diagrams, transcribes with Qwen3.6-35B-A3B VLM into 3-layer representation,
       and injects in-place into Markdown documents.
    """
    logger.info("=== STAGE 4: Image Template Filtering & Diagram Transcription ===")
    from processing.finalize_stage2_injection import (
        REPETITIVE_IMAGE_MAP,
        clean_repetitive_templates,
        inject_transcriptions_into_documents,
    )
    from processing.process_embedded_images import run_stage2_pipeline

    # 1. Run diagram identification and VLM transcription
    logger.info("Scanning for new candidate diagrams to transcribe...")
    run_stage2_pipeline()

    # 2. Clean templates and inject transcriptions into Markdown
    conn = open_db(db_path)
    cur = conn.cursor()

    cur.execute("SELECT id, markdown FROM documents WHERE markdown IS NOT NULL")
    docs = cur.fetchall()

    cleaned_templates = 0
    for did, md in docs:
        cleaned_md = clean_repetitive_templates(md, REPETITIVE_IMAGE_MAP)
        if cleaned_md != md:
            cleaned_templates += 1
            if not dry_run:
                cur.execute("UPDATE documents SET markdown = ? WHERE id = ?", (cleaned_md, did))

    if not dry_run and cleaned_templates > 0:
        conn.commit()

    conn.close()

    # Inject transcribed diagrams
    injected_count = inject_transcriptions_into_documents(db_path=db_path, dry_run=dry_run)
    logger.info("Stage 4 complete: %d docs cleaned of templates, %d diagrams injected.",
                cleaned_templates, injected_count)
    return {"cleaned_templates": cleaned_templates, "diagrams_injected": injected_count}


# ==============================================================================
# STAGE 5: HIERARCHICAL CHUNKING & VECTOR EMBEDDING
# ==============================================================================
def run_embedding_pipeline(
    db_path: str = DEFAULT_DB_PATH,
    concurrency: int = 16,
    batch_size: int = 16,
    skip_existing: bool = True,
) -> dict[str, int]:
    """
    Finds newly parsed or un-embedded documents:
    1. Hierarchically splits into Parent Chunks (~3000 chars) and Child Chunks (~500 chars).
    2. Generates 768-dim float32 embeddings for Child Chunks via FalcoLabs gateway.
    3. Indexes vectors into vec_chunks (sqlite-vec) and synchronizes fts_documents (FTS5).
    """
    logger.info("=== STAGE 5: Hierarchical Chunking & Vector Embedding ===")
    from processing.embeddings import run_hierarchical_chunk_embedding

    conn = open_db(db_path)
    cur = conn.cursor()
    # Check documents needing chunks
    cur.execute(
        """
        SELECT COUNT(*) FROM documents
        WHERE id NOT IN (SELECT DISTINCT doc_id FROM document_chunks)
          AND (id NOT IN (SELECT doc_id FROM document_stubs)
               OR id IN (SELECT doc_id FROM document_stubs WHERE status = 'extracted'))
          AND length(trim(markdown)) > 0
        """
    )
    unembedded_docs = cur.fetchone()[0]
    conn.close()

    logger.info("Documents requiring hierarchical chunking & embedding: %d", unembedded_docs)
    if unembedded_docs == 0 and skip_existing:
        logger.info("All documents are already embedded. Skipping.")
        return {"documents_embedded": 0, "child_chunks_embedded": 0}

    stats = run_hierarchical_chunk_embedding(
        db_path=db_path,
        embed_url=DEFAULT_EMBED_URL,
        model_name=DEFAULT_EMBED_MODEL,
        api_key=DEFAULT_EMBED_KEY,
        concurrency=concurrency,
        batch_size=batch_size,
        skip_existing=skip_existing,
        verbose=True,
    )

    # Synchronize USearch HNSW index
    from processing.build_usearch_index import build_usearch_index
    logger.info("Synchronizing USearch HNSW vector index at %s...", DEFAULT_INDEX_PATH)
    build_usearch_index(
        db_path=db_path,
        index_path=DEFAULT_INDEX_PATH,
        incremental=True,
    )

    return stats


# ==============================================================================
# STAGE 6: EXPORT UNSCRAPED LINKS TO CSV
# ==============================================================================
def export_unscraped_links(
    db_path: str = DEFAULT_DB_PATH,
    out_csv: str = "unscraped_ftu_links.csv",
) -> int:
    """
    Scans markdown and links metadata across all documents for FTU URLs
    that have never been crawled into documents.
    Exports them to CSV sorted by frequency.
    """
    logger.info("=== STAGE 6: Exporting Unscraped Candidate Links ===")
    conn = open_db(db_path)
    cur = conn.cursor()

    # Collect all scraped URLs
    cur.execute("SELECT id, url, redirected_url FROM documents")
    scraped = set()
    for did, u, ru in cur.fetchall():
        if u:
            scraped.add(normalize_url(u))
        if ru:
            scraped.add(normalize_url(ru))

    cur.execute("SELECT doc_id, url FROM document_stubs")
    for did, u in cur.fetchall():
        if u:
            scraped.add(normalize_url(u))

    unscraped: dict[str, dict[str, Any]] = {}

    # 1. From Markdown
    cur.execute("SELECT id, markdown FROM documents WHERE markdown IS NOT NULL")
    for did, md in cur.fetchall():
        for m in MD_LINK_REGEX.finditer(md):
            text, href = m.group(1).strip(), m.group(2).strip()
            norm = normalize_url(href)
            netloc = urlparse(norm).netloc
            if "ftu.edu.vn" in netloc and norm not in scraped:
                if norm not in unscraped:
                    unscraped[norm] = {
                        "url": norm,
                        "subdomain": netloc,
                        "occurrences": 0,
                        "sample_anchor_text": text,
                        "sample_source_doc_id": did,
                        "source": "markdown",
                    }
                unscraped[norm]["occurrences"] += 1
                if not unscraped[norm]["sample_anchor_text"] and text:
                    unscraped[norm]["sample_anchor_text"] = text

    # 2. From links metadata JSON
    cur.execute("SELECT id, links FROM documents WHERE links IS NOT NULL AND length(links) > 5")
    for did, links_json in cur.fetchall():
        try:
            data = json.loads(links_json)
            internal = data.get("internal", []) if isinstance(data, dict) else []
            for item in internal:
                href = item.get("href", "")
                text = item.get("text", "") or item.get("title", "")
                norm = normalize_url(href)
                netloc = urlparse(norm).netloc
                if "ftu.edu.vn" in netloc and norm not in scraped:
                    if norm not in unscraped:
                        unscraped[norm] = {
                            "url": norm,
                            "subdomain": netloc,
                            "occurrences": 0,
                            "sample_anchor_text": text,
                            "sample_source_doc_id": did,
                            "source": "links_metadata",
                        }
                    unscraped[norm]["occurrences"] += 1
                    if not unscraped[norm]["sample_anchor_text"] and text:
                        unscraped[norm]["sample_anchor_text"] = text
        except Exception:
            pass

    conn.close()

    sorted_links = sorted(unscraped.values(), key=lambda x: x["occurrences"], reverse=True)

    # Save to both target file and processing/ directory copy
    out_paths = [Path(out_csv)]
    proc_csv = PROJECT_ROOT / "processing" / "unscraped_links.csv"
    if proc_csv.resolve() != out_paths[0].resolve():
        out_paths.append(proc_csv)

    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["url", "subdomain", "occurrences", "sample_anchor_text", "sample_source_doc_id", "source"]
            )
            writer.writeheader()
            for row in sorted_links:
                writer.writerow(row)

    logger.info("Exported %d unique unscraped URLs (%d total references) to %s.",
                len(sorted_links), sum(x["occurrences"] for x in sorted_links), out_csv)
    return len(sorted_links)


# ==============================================================================
# PIPELINE STATUS DASHBOARD
# ==============================================================================
def get_pipeline_status(db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Inspect and report the current status of the knowledge base."""
    conn = open_db(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM documents")
    total_docs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM documents WHERE id IN (SELECT DISTINCT doc_id FROM document_chunks)")
    docs_with_chunks = cur.fetchone()[0]

    cur.execute("SELECT chunk_type, COUNT(*) FROM document_chunks GROUP BY chunk_type")
    chunk_counts = dict(cur.fetchall())

    cur.execute("SELECT COUNT(*) FROM vec_chunks")
    total_vectors = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM fts_documents")
    fts_docs = cur.fetchone()[0]

    cur.execute("SELECT status, COUNT(*) FROM document_stubs GROUP BY status")
    stubs_status = dict(cur.fetchall())

    # Pending items
    cur.execute("SELECT COUNT(*) FROM document_stubs WHERE status = 'pending'")
    pending_stubs = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*) FROM document_stubs
        WHERE (status = 'downloaded' OR (status = 'extracted' AND extracted_chars = 0))
          AND local_path IS NOT NULL
        """
    )
    pending_parse = cur.fetchone()[0]

    cur.execute(
        """
        SELECT COUNT(*) FROM documents
        WHERE id NOT IN (SELECT DISTINCT doc_id FROM document_chunks)
          AND (id NOT IN (SELECT doc_id FROM document_stubs)
               OR id IN (SELECT doc_id FROM document_stubs WHERE status = 'extracted'))
          AND length(trim(markdown)) > 0
        """
    )
    pending_embed = cur.fetchone()[0]

    conn.close()

    # USearch status
    usearch_file = Path(DEFAULT_INDEX_PATH)
    usearch_info = {"active": False, "keys": 0, "size_mb": 0.0}
    if usearch_file.exists() and usearch_file.stat().st_size > 0:
        usearch_info["size_mb"] = usearch_file.stat().st_size / (1024 * 1024)
        try:
            from usearch.index import Index, MetricKind, ScalarKind
            idx = Index(ndim=768, metric=MetricKind.Cos, dtype=ScalarKind.F16)
            idx.view(str(usearch_file))
            usearch_info["active"] = True
            usearch_info["keys"] = len(idx)
        except Exception:
            pass

    status = {
        "total_documents": total_docs,
        "documents_with_chunks": docs_with_chunks,
        "parent_chunks": chunk_counts.get("parent", 0),
        "child_chunks": chunk_counts.get("child", 0),
        "total_vectors": total_vectors,
        "fts_documents": fts_docs,
        "usearch": usearch_info,
        "stubs": stubs_status,
        "pending_work": {
            "stubs_to_fetch": pending_stubs,
            "files_to_parse": pending_parse,
            "docs_to_embed": pending_embed,
        },
    }
    return status


def print_status_dashboard(status: dict[str, Any]) -> None:
    """Print a clean dashboard to the console."""
    print("\n" + "=" * 65)
    print("         BABEL KNOWLEDGE BASE PIPELINE STATUS DASHBOARD")
    print("=" * 65)
    print(f"Total Documents in DB         : {status['total_documents']:,}")
    print(f"Documents Chunked & Embedded  : {status['documents_with_chunks']:,} / {status['total_documents']:,}")
    print(f"Total Parent Chunks           : {status['parent_chunks']:,}")
    print(f"Total Child Vectors Indexed   : {status['child_chunks']:,} (vec_chunks: {status['total_vectors']:,})")
    u_info = status.get("usearch", {})
    if u_info.get("active"):
        print(f"USearch HNSW Vector Index     : {u_info['keys']:,} keys ({u_info['size_mb']:.1f} MB, active)")
    else:
        print("USearch HNSW Vector Index     : Inactive (using sqlite-vec fallback)")
    print(f"FTS5 Search Indexed Documents : {status['fts_documents']:,}")
    print("-" * 65)
    print("Document Stubs Breakdown:")
    for st, count in status["stubs"].items():
        print(f"  • {st:<12}: {count:,}")
    print("-" * 65)
    print("Pending Pipeline Work (Delta):")
    pw = status["pending_work"]
    print(f"  • Pending Stubs to Fetch     : {pw['stubs_to_fetch']:,}")
    print(f"  • Downloaded Files to Parse  : {pw['files_to_parse']:,}")
    print(f"  • Documents to Chunk & Embed : {pw['docs_to_embed']:,}")
    print("=" * 65 + "\n")


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def run_all_stages(
    db_path: str = DEFAULT_DB_PATH,
    concurrency_embed: int = 16,
    concurrency_parse: int = 4,
    export_csv: bool = True,
) -> None:
    """Run the entire end-to-end periodic processing loop."""
    t0 = time.time()
    logger.info("Starting complete periodic processing pipeline...")

    # Stage 1: Links & Stubs
    run_link_canonicalization_and_stubs(db_path=db_path)

    # Stage 2: Fetch Stubs
    run_fetch_stubs(db_path=db_path)

    # Stage 3: Parse & OCR
    run_parse_documents(db_path=db_path, concurrency=concurrency_parse)

    # Stage 4: Images & Diagrams
    run_image_diagram_pipeline(db_path=db_path)

    # Stage 5: Hierarchical Chunking & Embedding
    run_embedding_pipeline(db_path=db_path, concurrency=concurrency_embed)

    # Stage 6: Export Unscraped Links
    if export_csv:
        export_unscraped_links(db_path=db_path)

    # Final Dashboard
    st = get_pipeline_status(db_path=db_path)
    print_status_dashboard(st)
    logger.info("Periodic pipeline run completed successfully in %.1f seconds.", time.time() - t0)


def main():
    parser = argparse.ArgumentParser(
        description="Streamlined Periodic Processing Pipeline for Babel Knowledge Base."
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to stash.db")
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline (all stages)")
    parser.add_argument("--status", action="store_true", help="Display knowledge base status dashboard")
    parser.add_argument("--links", action="store_true", help="Run Stage 1: Link canonicalization & stubbing")
    parser.add_argument("--fetch", action="store_true", help="Run Stage 2: Fetch pending document stubs")
    parser.add_argument("--parse", action="store_true", help="Run Stage 3: Parse documents & GPU OCR")
    parser.add_argument("--images", action="store_true", help="Run Stage 4: Image template filtering & diagram VLM")
    parser.add_argument("--embed", action="store_true", help="Run Stage 5: Hierarchical chunking & embedding")
    parser.add_argument("--export-unscraped", action="store_true", help="Run Stage 6: Export unscraped links to CSV")
    parser.add_argument("--concurrency-embed", type=int, default=16, help="Embedding concurrency (default: 16)")
    parser.add_argument("--concurrency-parse", type=int, default=4, help="Parsing concurrency (default: 4)")
    parser.add_argument("--force", action="store_true", help="Force re-processing of items")

    args = parser.parse_args()

    # Default action if no stage specified is --status
    if not any([args.all, args.status, args.links, args.fetch, args.parse, args.images, args.embed, args.export_unscraped]):
        args.status = True

    if args.status:
        st = get_pipeline_status(args.db_path)
        print_status_dashboard(st)

    if args.links:
        run_link_canonicalization_and_stubs(args.db_path)

    if args.fetch:
        run_fetch_stubs(args.db_path, force=args.force)

    if args.parse:
        run_parse_documents(args.db_path, concurrency=args.concurrency_parse, force=args.force)

    if args.images:
        run_image_diagram_pipeline(args.db_path)

    if args.embed:
        run_embedding_pipeline(args.db_path, concurrency=args.concurrency_embed, skip_existing=not args.force)

    if args.export_unscraped:
        export_unscraped_links(args.db_path)

    if args.all:
        run_all_stages(
            db_path=args.db_path,
            concurrency_embed=args.concurrency_embed,
            concurrency_parse=args.concurrency_parse,
        )


if __name__ == "__main__":
    main()
