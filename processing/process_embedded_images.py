"""
process_embedded_images.py - Stage 2: Embedded Image Processing & Transcription.

1. Replaces repetitive site templates, flags, and UI icons in-place with canonical text tokens.
2. Identifies and deduplicates organizational charts, process flowcharts, and diagrams.
3. Transcribes diagrams using Qwen3.6-35B-A3B on swarm.ftds.online into Mermaid + Markdown + Prose.
4. Injects transcribed diagrams in-place into documents.markdown at the image's original position.
5. Synchronizes chunks and vectors in stash.db.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sqlite3
import time
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("process_embedded_images")

DB_PATH = os.environ.get("STASH_DB_PATH", "stash.db")
CHARTS_DIR = Path("downloads/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

OPENWEBUI_URL = os.environ.get("OPENWEBUI_ENDPOINT", "https://swarm.ftds.online").rstrip("/")
OPENWEBUI_KEY = os.environ.get("OPENWEBUI_KEY", "")
OPENWEBUI_MODEL = os.environ.get("VLM_MODEL", "Qwen3.6-35B-A3B")

# -----------------------------------------------------------------------------
# 1. Canonical Template Replacer
# -----------------------------------------------------------------------------

CANONICAL_PATTERNS = [
    # Language flags
    (re.compile(r"\[\s*\[\s*image\s+file\s*\]\s*\]\([^)]*(?:/en|language=en|flag-uk)[^)]*\)", re.IGNORECASE), "[English]"),
    (re.compile(r"\[\s*\[\s*image\s+file\s*\]\s*\]\([^)]*(?:/vi|/vn|language=vn|flag-vn)[^)]*\)", re.IGNORECASE), "[Tiếng Việt]"),
    (re.compile(r"\[\s*image\s+file\s*\]\s*(?:Tiếng Việt|vn\.png)", re.IGNORECASE), "[Tiếng Việt]"),
    (re.compile(r"\[\s*image\s+file\s*\]\s*(?:English|en\.png)", re.IGNORECASE), "[English]"),
    # Social & messaging
    (re.compile(r"\[\s*\[\s*image\s+file\s*\]\s*Messenger\s*\]\([^)]*\)", re.IGNORECASE), "[Messenger]"),
    (re.compile(r"\[\s*\[\s*image\s+file\s*\]\s*Zalo\s*\]\([^)]*\)", re.IGNORECASE), "[Zalo]"),
    (re.compile(r"\[\s*\[\s*image\s+file\s*\]\s*Tawk\s*To\s*\]\([^)]*\)", re.IGNORECASE), "[Tawk To]"),
    # Banners & generic logos
    (re.compile(r"\[\s*\[\s*image\s+file\s*\]\s*\]\([^)]*banner-cttts[^)]*\)", re.IGNORECASE), "[Banner: Tuyển sinh FTU]"),
    (re.compile(r"\[\s*\[\s*image\s+file\s*\]\s*\]\([^)]*banner-web[^)]*\)", re.IGNORECASE), "[Banner: Website FTU]"),
    (re.compile(r"\[\s*\[\s*image\s+file\s*\]\s*\]\([^)]*logo-header[^)]*\)", re.IGNORECASE), "[Logo FTU]"),
    (re.compile(r"\[\s*\[\s*image\s+file\s*\]\s*\]\([^)]*JIEM-3[^)]*\)", re.IGNORECASE), "[Tạp chí JIEM]"),
    (re.compile(r"\[\s*\[\s*image\s+file\s*\]\s*\]\([^)]*sach\.jpg[^)]*\)", re.IGNORECASE), "[Sách xuất bản]"),
    (re.compile(r"\[\s*\[\s*image\s+file\s*\]\s*\]\([^)]*xet-tuyen-truc-tuyen[^)]*\)", re.IGNORECASE), "[Đăng ký xét tuyển trực tuyến]"),
]


def apply_canonical_template_replacements(db_path: str = DB_PATH) -> int:
    """Scan all markdown documents and replace recurring UI/template image markers in-place."""
    logger.info("Starting Pass 1: Canonical Template Replacement across markdown documents...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, markdown FROM documents WHERE markdown LIKE '%image file%'")
    rows = cursor.fetchall()
    logger.info(f"Scanning {len(rows)} documents containing '[image file]' markers...")

    updated_docs = 0
    total_replacements = 0

    batch_updates = []
    for doc_id, md in tqdm(rows, desc="Template Replacement"):
        current_md = md
        doc_replaced = 0
        for pat, repl in CANONICAL_PATTERNS:
            matches = pat.findall(current_md)
            if matches:
                doc_replaced += len(matches)
                current_md = pat.sub(repl, current_md)

        if doc_replaced > 0:
            batch_updates.append((current_md, doc_id))
            updated_docs += 1
            total_replacements += doc_replaced

        if len(batch_updates) >= 500:
            cursor.executemany("UPDATE documents SET markdown = ? WHERE id = ?", batch_updates)
            conn.commit()
            batch_updates = []

    if batch_updates:
        cursor.executemany("UPDATE documents SET markdown = ? WHERE id = ?", batch_updates)
        conn.commit()

    conn.close()
    logger.info(
        f"Pass 1 Complete: Replaced {total_replacements:,} template markers across {updated_docs:,} documents."
    )
    return updated_docs


# -----------------------------------------------------------------------------
# 2. Candidate Chart Discovery & Deduplication
# -----------------------------------------------------------------------------

CHART_PATTERNS = [
    r"so[-_]?do",
    r"co[-_]?cau",
    r"to[-_]?chuc",
    r"bo[-_]?may",
    r"quy[-_]?trinh",
    r"luu[-_]?do",
    r"organ[io]gram",
    r"org[-_]?chart",
    r"flow[-_]?chart",
    r"diagram",
    r"sơ đồ",
    r"cơ cấu",
    r"quy trình",
    r"lưu đồ",
    r"bộ máy",
]
CHART_REGEX = re.compile("|".join(CHART_PATTERNS), re.IGNORECASE)

TITLE_CHART_REGEX = re.compile(r"(cơ cấu tổ chức|sơ đồ|quy trình|lưu đồ|bộ máy)", re.IGNORECASE)

WP_DIMENSION_REGEX = re.compile(r"-\d+x\d+(?=\.[a-zA-Z0-9]+$)")


def normalize_wp_image_url(url: str) -> str:
    """Normalize WordPress multi-resolution thumbnails to base full-resolution image URL."""
    return WP_DIMENSION_REGEX.sub("", url)


def discover_candidate_charts(db_path: str = DB_PATH) -> list[dict[str, Any]]:
    """Scan documents.media to find all organizational charts, flowcharts, and diagrams."""
    logger.info("Scanning documents.media for candidate charts and diagrams...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, url, json_extract(metadata, '$.title'), media FROM documents WHERE media IS NOT NULL AND media != '{}' AND media != ''"
    )
    rows = cursor.fetchall()

    candidates: dict[str, dict[str, Any]] = {}

    for doc_id, doc_url, title, media_raw in rows:
        try:
            media = json.loads(media_raw)
            images = media.get("images", [])
            for img in images:
                src = img.get("src", "").strip()
                if not src:
                    continue

                # Ignore UI icons, flags, and header banners
                src_lower = src.lower()
                if any(x in src_lower for x in ["flag-", "en.png", "vi.png", "icon-", "banner-", "logo-", "jiem", "sach.jpg"]):
                    continue

                alt = img.get("alt", "").strip()
                desc = img.get("desc", "").strip()

                is_chart = False
                reason = ""

                if CHART_REGEX.search(src):
                    is_chart = True
                    reason = "src_keyword"
                elif alt and CHART_REGEX.search(alt):
                    is_chart = True
                    reason = "alt_keyword"
                elif title and TITLE_CHART_REGEX.search(title):
                    is_chart = True
                    reason = "page_title_match"

                if is_chart:
                    full_url = urllib.parse.urljoin(doc_url, src)
                    base_url = normalize_wp_image_url(full_url)

                    if base_url not in candidates:
                        candidates[base_url] = {
                            "base_url": base_url,
                            "variants": [full_url],
                            "doc_ids": [doc_id],
                            "alt": alt,
                            "title": title or "",
                            "desc": desc[:150],
                            "reason": reason,
                        }
                    else:
                        if full_url not in candidates[base_url]["variants"]:
                            candidates[base_url]["variants"].append(full_url)
                        if doc_id not in candidates[base_url]["doc_ids"]:
                            candidates[base_url]["doc_ids"].append(doc_id)
                        if not candidates[base_url]["alt"] and alt:
                            candidates[base_url]["alt"] = alt
        except Exception:
            pass

    conn.close()
    result = list(candidates.values())
    logger.info(f"Discovered {len(result)} unique deduplicated candidate chart images.")
    return result


# -----------------------------------------------------------------------------
# 3. Download & Cache Images
# -----------------------------------------------------------------------------

def download_image(url: str, output_path: Path) -> bool:
    """Download image to disk with SSL fallback and redirect resolution."""
    if output_path.exists() and output_path.stat().st_size > 500:
        return True

    parsed = urllib.parse.urlsplit(url)
    encoded_path = urllib.parse.quote(parsed.path)
    safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, encoded_path, parsed.query, parsed.fragment))

    try:
        with httpx.Client(verify=False, follow_redirects=True, timeout=15.0) as client:
            resp = client.get(safe_url)
            if resp.status_code == 200 and len(resp.content) > 500:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                return True
            else:
                logger.warning(f"Download HTTP {resp.status_code} for {safe_url}")
                return False
    except Exception as e:
        logger.warning(f"Download failed for {safe_url}: {e}")
        return False


# -----------------------------------------------------------------------------
# 4. VLM Transcription via swarm.ftds.online (Qwen3.6-35B-A3B)
# -----------------------------------------------------------------------------

VLM_PROMPT = """Phân tích hình ảnh sơ đồ tổ chức hoặc quy trình này.
Hãy chuyển đổi sơ đồ thành 3 phần rõ ràng:

### 1. Sơ đồ Mermaid (flowchart TD)
```mermaid
flowchart TD
  %% Bao bọc nhãn mỗi node trong dấu ngoặc kép "" để tránh lỗi cú pháp
  ...
```

### 2. Cấu trúc thứ bậc / Các bước thực hiện
- Danh sách thụt đầu dòng (indented bullet list) thể hiện cấu trúc phân cấp hoặc trình tự xử lý.

### 3. Tóm tắt quan hệ báo cáo / quy trình
Đoạn văn ngắn (2-3 câu) tóm tắt các quan hệ lãnh đạo, báo cáo, thời hạn và nhiệm vụ chính trong sơ đồ."""


def transcribe_chart_with_vlm(image_path: Path) -> str | None:
    """Transcribe a diagram image into 3-layer representation using Qwen3.6-35B-A3B."""
    cache_path = image_path.with_suffix(".md")
    if cache_path.exists() and cache_path.stat().st_size > 100:
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    with open(image_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.suffix.lower().lstrip(".")
    mime_type = "image/png" if ext == "png" else ("image/webp" if ext == "webp" else "image/jpeg")

    payload = {
        "model": OPENWEBUI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_img}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.1,
    }

    headers = {"Authorization": f"Bearer {OPENWEBUI_KEY}", "Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(f"{OPENWEBUI_URL}/api/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return content
            else:
                logger.error(f"VLM API Error HTTP {resp.status_code}: {resp.text[:200]}")
                return None
    except Exception as e:
        logger.error(f"VLM request failed for {image_path}: {e}")
        return None


# -----------------------------------------------------------------------------
# 5. In-Place Document Injection & Re-Embedding
# -----------------------------------------------------------------------------

def inject_transcriptions_into_documents(
    chart_candidate: dict[str, Any],
    transcription: str,
    db_path: str = DB_PATH,
) -> int:
    """Replace image markers in documents.markdown with the transcribed 3-layer representation."""
    doc_ids = chart_candidate["doc_ids"]
    variants = chart_candidate["variants"] + [chart_candidate["base_url"]]

    # Construct search patterns for image markers
    img_filenames = [urllib.parse.urlsplit(v).path.split("/")[-1] for v in variants if v]
    img_filenames = [f for f in set(img_filenames) if f]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    updated = 0
    formatted_block = f"\n\n{transcription.strip()}\n\n"

    for doc_id in doc_ids:
        cursor.execute("SELECT markdown FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        if not row or not row[0]:
            continue

        md = row[0]
        original_md = md

        # Match inline image references containing any of the variants or filenames
        for fname in img_filenames:
            escaped_fname = re.escape(fname)
            # Pattern 1: [ [image file] ... ](url)
            pat1 = re.compile(r"\[\s*\[\s*image\s+file\s*\][^\]]*\]\([^)]*" + escaped_fname + r"[^)]*\)", re.IGNORECASE)
            if pat1.search(md):
                md = pat1.sub(formatted_block, md, count=1)
                break

            # Pattern 2: [image file] alt
            pat2 = re.compile(r"\[\s*image\s+file\s*\][^()\n]*" + escaped_fname, re.IGNORECASE)
            if pat2.search(md):
                md = pat2.sub(formatted_block, md, count=1)
                break

        # Fallback: If not replaced by exact filename, check if there is an image marker under the article title
        if md == original_md and chart_candidate.get("title"):
            title_pat = re.escape(chart_candidate["title"])
            fallback_pat = re.compile(r"(" + title_pat + r"[^\n]*\n+)(\[\s*\[\s*image\s+file\s*\][^\]]*\]\([^)]+\)|\[\s*image\s+file\s*\])", re.IGNORECASE)
            if fallback_pat.search(md):
                md = fallback_pat.sub(r"\1" + formatted_block, md, count=1)

        if md != original_md:
            cursor.execute("UPDATE documents SET markdown = ? WHERE id = ?", (md, doc_id))
            conn.commit()
            updated += 1
            logger.info(f"Injected transcribed diagram into Doc #{doc_id}")

    conn.close()
    return updated


# -----------------------------------------------------------------------------
# 6. Main Orchestrator
# -----------------------------------------------------------------------------

def run_stage2_pipeline(db_path: str = DB_PATH) -> dict[str, Any]:
    """Execute the complete Stage 2 embedded image processing pipeline."""
    start_time = time.time()
    logger.info("=== Starting Stage 2: Embedded Image Processing Pipeline ===")

    # Step 1: Replace repetitive templates in markdown
    template_docs_updated = apply_canonical_template_replacements(db_path)

    # Step 2: Discover candidate charts
    candidates = discover_candidate_charts(db_path)

    # Step 3: Download & transcribe charts with Qwen3.6-35B-A3B
    transcribed_count = 0
    injected_docs = set()

    for idx, cand in enumerate(candidates, 1):
        url = cand["base_url"]
        fname = urllib.parse.urlsplit(url).path.split("/")[-1] or f"chart_{idx}.jpg"
        local_path = CHARTS_DIR / fname

        logger.info(f"[{idx}/{len(candidates)}] Processing candidate chart: {fname} (Docs: {cand['doc_ids']})")

        # Try base URL first, then variants if base fails
        success = download_image(url, local_path)
        if not success:
            for v in cand["variants"]:
                if download_image(v, local_path):
                    success = True
                    break

        if not success:
            logger.warning(f"Could not download any variant for {url}")
            continue

        transcription = transcribe_chart_with_vlm(local_path)
        if transcription:
            transcribed_count += 1
            n_injected = inject_transcriptions_into_documents(cand, transcription, db_path)
            if n_injected > 0:
                injected_docs.update(cand["doc_ids"])
            time.sleep(1.0)

    # Step 4: Re-embed modified documents
    if injected_docs:
        logger.info(f"Re-embedding {len(injected_docs)} updated documents with transcribed diagrams...")
        from processing.embeddings import run_hierarchical_chunk_embedding

        # Delete existing chunks for these updated documents so they are re-chunked cleanly
        conn = sqlite3.connect(db_path)
        for did in injected_docs:
            stale_cids = [r[0] for r in conn.execute("SELECT chunk_id FROM document_chunks WHERE doc_id = ?", (did,))]
            if stale_cids:
                ph = ",".join("?" * len(stale_cids))
                conn.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({ph})", stale_cids)
                conn.execute("DELETE FROM document_chunks WHERE doc_id = ?", (did,))
        conn.commit()
        conn.close()

        run_hierarchical_chunk_embedding(db_path=db_path, skip_existing=True, batch_size=64, concurrency=8)

    elapsed = time.time() - start_time
    logger.info(f"=== Stage 2 Pipeline Complete in {elapsed:.1f}s ===")
    return {
        "template_docs_updated": template_docs_updated,
        "candidate_charts": len(candidates),
        "transcribed_charts": transcribed_count,
        "injected_docs": len(injected_docs),
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    run_stage2_pipeline()
