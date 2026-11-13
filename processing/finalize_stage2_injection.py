"""
finalize_stage2_injection.py - Precise in-place diagram injection and re-embedding.

Injects transcribed Mermaid flowcharts and hierarchical structures into target documents
at the exact position of the original image placeholders, followed by re-chunking,
vector embedding, and FTS5 synchronization.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import sys
from pathlib import Path
import sqlite_vec

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.embeddings import run_hierarchical_chunk_embedding
from server.services.stash import Stash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finalize_stage2")

DB_PATH = os.environ.get("STASH_DB_PATH", "stash.db")
CHARTS_DIR = Path("downloads/charts")


def open_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

# Specific diagram-to-document injection definitions
INJECTIONS = [
    {
        "chart_md": "dang-ky-tin-chi-1.md",
        "doc_ids": [1653],
        "anchor_regex": r"(# HƯỚNG DẪN QUY TRÌNH ĐĂNG KÝ TÍN CHỈ HỌC TẬP[\s\S]*?Phòng Quản lý Đào tạo hướng dẫn[^\n]*\n+)(\[image file\])",
        "fallback_regex": r"(\[image file\])",
    },
    {
        "chart_md": "quy-trinh-giai-dap-thac-12032021095557.md",
        "doc_ids": [1751],
        "anchor_regex": r"(## QUY TRÌNH GIẢI ĐÁP THẮC MẮC[^\n]*\n+)(\[image file\])",
        "fallback_regex": r"(\[image file\])",
    },
    {
        "chart_md": "quy-trinh-giai-dap-thac-12032021095557.md",
        "doc_ids": [2],
        "anchor_regex": r"(\[\s*\[\s*image\s+file\s*\]\s*QUY TRÌNH GIẢI ĐÁP THẮC MẮC[^\n]*?\]\([^)]+\))",
    },
    {
        "chart_md": "So-do-phong-Thi-dot-1.2025-T.md",
        "doc_ids": [8050],
        "anchor_regex": r"(\*\*5\\?\.\s*Sơ đồ phòng xét tuyển:\*\*\n+)(\[image file\])",
        "fallback_regex": r"(\*\*5\\?\.\s*Sơ đồ phòng xét tuyển:\*\*)",
    },
    {
        "chart_md": "So-do-phong-thi.md",
        "doc_ids": [8075],
        "anchor_regex": r"(\*\*6\\?\.\s*Sơ đồ phòng xét tuyển:\*\*\n+)(\[image file\])",
        "fallback_regex": r"(\*\*6\\?\.\s*Sơ đồ phòng xét tuyển:\*\*)",
    },
    {
        "chart_md": "so-do-phong-thi.md",
        "doc_ids": [1941, 1965],
        "anchor_regex": r"(\[.*?so[-_]?do[-_]?phong[-_]?thi.*?\]\([^)]+\)|\[image file\])",
    },
    {
        "chart_md": "ap-dung-quy-trinh-pdca-trong-quan-ly-chat-luong-1.md",
        "doc_ids": [8049],
        "anchor_regex": r"(###\s+\[\s*Áp dụng quy trình PDCA[^\n]*\n+[^\n]*\n+)(\[\s*\[\s*image\s+file\s*\]\s*\]\([^)]+\))",
    },
    {
        "chart_md": "co-cau-doan-thanh-nien-2024-2027-1.md",
        "doc_ids": [8090, 8154],
        "anchor_regex": r"(\[.*?co[-_]?cau[-_]?doan[-_]?thanh[-_]?nien.*?\]\([^)]+\)|\[\s*\[\s*image\s+file\s*\][^\n]*co[-_]?cau[^\n]*)",
    },
    {
        "chart_md": "Thong-bao-to-chuc-thi-VSTEP-Dot-4.-Ngay-thi-0910-thang-5.2026.md",
        "doc_ids": [15, 1925, 1964, 2717, 2718],
        "anchor_regex": r"(\[.*?Thong[-_]?bao[-_]?to[-_]?chuc[-_]?thi[-_]?VSTEP.*?\]\([^)]+\)|\[\s*\[\s*image\s+file\s*\][^\n]*VSTEP[^\n]*)",
    },
    {
        "chart_md": "thong-bao-so-do-su-14112025014757.md",
        "doc_ids": [6777],
        "anchor_regex": r"(#### \[THÔNG BÁO SƠ ĐỒ SỰ KIỆN[^\n]*?\]\([^)]+\))",
    },
    {
        "chart_md": "Dang-ky-bo-sung-CTTD.md",
        "doc_ids": [8186, 8210, 8211, 8216],
        "anchor_regex": r"(\[.*?THÔNG BÁO VỀ VIỆC ĐĂNG KÝ BỔ SUNG CHƯƠNG TRÌNH TRAO ĐỔI[^\n]*?\]\([^)]+\))",
    },
]


def execute_injections(db_path: str = DB_PATH) -> set[int]:
    """Perform targeted in-place diagram injections into markdown documents."""
    conn = open_db(db_path)
    cursor = conn.cursor()

    modified_docs: set[int] = set()

    for item in INJECTIONS:
        chart_file = CHARTS_DIR / item["chart_md"]
        if not chart_file.exists():
            logger.warning(f"Chart file missing: {chart_file}")
            continue

        transcription = chart_file.read_text(encoding="utf-8").strip()
        formatted_block = f"\n\n{transcription}\n\n"

        for doc_id in item["doc_ids"]:
            cursor.execute("SELECT markdown FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if not row or not row[0]:
                continue

            md = row[0]
            if "flowchart TD" in md and item["chart_md"] in md:
                logger.info(f"Doc #{doc_id} already contains {item['chart_md']}, skipping.")
                modified_docs.add(doc_id)
                continue

            orig_md = md
            # Try primary anchor regex
            pat = re.compile(item["anchor_regex"], re.IGNORECASE)
            match = pat.search(md)

            if match:
                # If group 2 exists (e.g. replacing placeholder under header)
                if match.lastindex and match.lastindex >= 2:
                    md = pat.sub(r"\1" + formatted_block, md, count=1)
                else:
                    md = pat.sub(match.group(1) + formatted_block, md, count=1)
            elif "fallback_regex" in item:
                fpat = re.compile(item["fallback_regex"], re.IGNORECASE)
                if fpat.search(md):
                    md = fpat.sub(r"\1" + formatted_block, md, count=1)

            if md != orig_md:
                cursor.execute("UPDATE documents SET markdown = ? WHERE id = ?", (md, doc_id))
                conn.commit()
                modified_docs.add(doc_id)
                logger.info(f"Successfully injected {item['chart_md']} into Doc #{doc_id}")
            else:
                logger.warning(f"Could not anchor {item['chart_md']} in Doc #{doc_id}")

    conn.close()
    return modified_docs


def reembed_and_sync(modified_docs: set[int], db_path: str = DB_PATH):
    """Re-chunk, re-embed, and sync FTS5 for all updated documents."""
    if not modified_docs:
        logger.info("No documents to re-embed.")
        return

    logger.info(f"Purging old chunks for {len(modified_docs)} modified documents...")
    conn = open_db(db_path)
    cursor = conn.cursor()

    for did in modified_docs:
        cids = [r[0] for r in cursor.execute("SELECT chunk_id FROM document_chunks WHERE doc_id = ?", (did,)).fetchall()]
        if cids:
            ph = ",".join("?" * len(cids))
            cursor.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({ph})", cids)
            cursor.execute("DELETE FROM document_chunks WHERE doc_id = ?", (did,))

        # Sync FTS5
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

    logger.info(f"Re-embedding {len(modified_docs)} documents with hierarchical embeddings...")
    run_hierarchical_chunk_embedding(db_path=db_path, skip_existing=True, batch_size=64, concurrency=8)
    logger.info("Re-embedding complete!")


def verify_diagram_queries():
    """Verify that transcribed diagrams are accurately retrieved via hybrid search."""
    logger.info("=== Verifying Hybrid Search on Injected Diagrams ===")
    stash = Stash(db_path=DB_PATH)

    test_queries = [
        ("Khoa Sau đại học", "Ai phụ trách quản lý đào tạo thạc sĩ liên kết Khoa Sau đại học?"),
        ("Khiếu nại sinh viên KĐTQT", "Quy trình giải đáp thắc mắc khiếu nại của sinh viên Khoa Đào tạo Quốc tế có mấy cấp?"),
        ("Đăng ký tín chỉ", "Quy trình đăng ký tín chỉ học tập Phòng Quản lý Đào tạo gồm mấy bước?"),
        ("Sơ đồ phòng thi", "Sơ đồ bố trí phòng xét tuyển tuyển sinh thạc sĩ Đợt 1 năm 2025"),
        ("Quy trình PDCA", "Áp dụng chu trình PDCA và chuẩn hóa trong quản lý chất lượng"),
    ]

    for category, query in test_queries:
        logger.info(f"\n--- Testing [{category}]: '{query}' ---")
        results = stash.search_hybrid(query, top_k=3)
        for idx, res in enumerate(results, 1):
            title = res.get("title") or "Untitled"
            score = res.get("score", 0.0)
            doc_id = res.get("doc_id")
            snippet = res.get("snippet", "").replace("\n", " ")[:160]
            logger.info(f"  Rank {idx} | Doc #{doc_id} | Score: {score:.4f} | Title: {title}")
            logger.info(f"         Snippet: {snippet}...")


if __name__ == "__main__":
    updated = execute_injections()
    # Add doc 1934 and 7130 which were already injected in Pass 2
    updated.update([1934, 7130])
    logger.info(f"Total updated documents: {len(updated)}: {sorted(list(updated))}")
    reembed_and_sync(updated)
    verify_diagram_queries()
