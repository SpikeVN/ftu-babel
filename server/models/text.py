"""Text cleaning and formatting utilities for Babel."""

from __future__ import annotations

import html
import re
import urllib.parse
from typing import Any, Sequence


def clean_normal_text(text: str) -> str:
    """Strip out markdown headers, images, links, raw URLs, and markup so only normal text remains."""
    if not text:
        return ""

    # 1. Unescape HTML entities
    text = html.unescape(text)

    # 2. Strip images: ![alt](url "title"), ![alt][ref], <img ...>, [image file], etc.
    text = re.sub(r"!\[.*?\](?:\([^)]*\)|\[[^\]]*\])?", "", text)
    text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\[?\s*image(?:\s+file)?\s*\]?\s*\](?:\([^)]*\))?", "", text, flags=re.IGNORECASE)

    # 3. Strip link reference definitions: [ref]: http://...
    text = re.sub(r"^[ \t]*\[[^\]]+\]:\s*\S+.*$", "", text, flags=re.MULTILINE)

    # 4. Strip HTML links keeping anchor text
    text = re.sub(r"<a\s+[^>]*>(.*?)</a>", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
    # Strip any remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 5. Markdown links: [anchor text](url) -> anchor text
    def _link_replace(m: re.Match) -> str:
        inner = m.group(1).strip()
        # If anchor text is a URL or empty, remove it completely
        if re.match(r"^(?:https?://|ftp://|www\.)", inner, flags=re.IGNORECASE):
            return ""
        return inner

    for _ in range(3):
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", _link_replace, text)
        text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", _link_replace, text)

    # Truncated or broken markdown link remnants
    text = re.sub(r"\[([^\]]+)\]\([^)]*$", _link_replace, text)
    text = re.sub(r"\[([^\]\n]+)$", r"\1", text)
    text = re.sub(r"\]\([^)]*\)?", "", text)

    # 6. Strip all raw URLs (http, https, ftp, www)
    text = re.sub(r"https?://\S+|ftp://\S+|www\.[a-zA-Z0-9.\-_]+\.[a-zA-Z]{2,}\S*", "", text)

    # 7. Strip markdown headers: #, ##, ###, etc. at start of lines or string
    text = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*[=\-]{2,}[ \t]*$", "", text, flags=re.MULTILINE)

    # 8. Strip markdown list bullets, blockquotes
    text = re.sub(r"^[ \t]*[*\-+][ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\d+\.[ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*>[ \t]*", "", text, flags=re.MULTILINE)

    # 9. Markdown escapes: \-, \*, etc.
    text = re.sub(r"\\([\\`*{}\[\]()#+\-.!_>])", r"\1", text)

    # 10. Strip bold, italic, strikethrough, backtick formatting
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # 11. Clean standalone __ or multiple underscores
    text = re.sub(r"^[ \t]*_{2,}[ \t]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]*_{2,}[ \t]*", " ", text)

    # 12. Normalize whitespace per line, filter blank lines
    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if re.match(r"^[\s_\-*.~,;:|/\\]*$", line):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


_BOILERPLATE_LINE_RE = re.compile(
    r"^(?:chi tiết|next article.*|prev(?:ious)? article.*|giới thiệu|tin tức|trang chủ)$",
    flags=re.IGNORECASE,
)


def extract_title_from_text(text: str, url: str = "") -> str:
    """Extract a clean title from the first markdown heading, or fall back to the URL slug."""
    if text:
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#"):
                heading = clean_normal_text(s.lstrip("#").strip())
                if heading and not _BOILERPLATE_LINE_RE.match(heading):
                    return heading[:200]
    if url:
        decoded = urllib.parse.unquote(url).rstrip("/")
        slug = decoded.rsplit("/", 1)[-1]
        slug = re.sub(r"^\d+-", "", slug)
        slug = re.sub(r"[-_]+", " ", slug).strip()
        if slug:
            return slug[:200]
    return ""


def clean_passage_snippet(text: str, max_chars: int = 900) -> str:
    """Clean passage text and strip common CMS boilerplate lines before truncating."""
    cleaned = clean_normal_text(text)
    if not cleaned:
        return ""
    kept: list[str] = []
    for line in cleaned.splitlines():
        s = line.strip()
        if not s or _BOILERPLATE_LINE_RE.match(s) or s in ("(", ")", "[", "]"):
            continue
        kept.append(s)
    joined = "\n".join(kept).strip() or cleaned
    if len(joined) > max_chars:
        return joined[:max_chars].rstrip() + "..."
    return joined


def format_search_results_as_text(results: Sequence[Any]) -> str:
    """Format search results into clean, token-efficient plain text for RAG LLM ingestion."""
    if not results:
        return "No results found."

    blocks = []
    for idx, r in enumerate(results, 1):
        doc_id = getattr(r, "doc_id", None) or (r.get("doc_id") if isinstance(r, dict) else "")
        title = getattr(r, "title", None) or (r.get("title") if isinstance(r, dict) else "")
        url = getattr(r, "url", None) or (r.get("url") if isinstance(r, dict) else "")
        score = getattr(r, "score", None) or (r.get("score") if isinstance(r, dict) else 0.0)
        dt = getattr(r, "date", None) or (r.get("date") if isinstance(r, dict) else "")
        summary = getattr(r, "summary", None) or (r.get("summary") if isinstance(r, dict) else "")
        content = getattr(r, "content", None) or (r.get("content") or r.get("snippet") if isinstance(r, dict) else "")

        score_val = float(score) if score is not None else 0.0
        cite_label = (title or extract_title_from_text("", str(url)) or str(url)).replace("[", "(").replace("]", ")")
        lines = [
            f"[{idx}] Document ID: {doc_id} (Score: {score_val:.4f})",
        ]
        if title:
            lines.append(f"Title: {title}")
        lines.append(f"URL: {url}")
        if url:
            lines.append(f"Cite: [{cite_label}]({url})")
        if dt:
            lines.append(f"Date: {dt}")
        if summary:
            lines.append(f"Summary (auto-generated): {summary}")
        if content:
            brief = content.rstrip()
            if not brief.endswith("..."):
                brief += "..."
            lines.append(f"Brief Content (truncated):\n{brief}")
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)


def format_documents_as_text(documents: Sequence[dict[str, Any]]) -> str:
    """Format full document details into clean plain text for RAG ingestion."""
    if not documents:
        return "No documents found."

    blocks = []
    for doc in documents:
        lines = [
            f"Title: {doc.get('title', '')}",
            f"Document ID: {doc.get('doc_id', '')}",
            f"URL: {doc.get('url', '')}",
        ]
        if doc.get("date"):
            lines.append(f"Date: {doc['date']}")
        if doc.get("summary"):
            lines.append(f"Summary: {doc['summary']}")
        header_text = "\n".join(lines)
        content = doc.get("content", "").strip()
        blocks.append(f"{header_text}\n\n{content}")

    return "\n\n========================================\n\n".join(blocks)


def format_web_search_results_as_text(results: Sequence[Any]) -> str:
    """Format web search results into clean plain text for RAG LLM ingestion."""
    if not results:
        return "No web search results found."

    blocks = []
    for idx, r in enumerate(results, 1):
        title = getattr(r, "title", None) or (r.get("title") if isinstance(r, dict) else "")
        url = getattr(r, "url", None) or (r.get("url") if isinstance(r, dict) else "")
        content = getattr(r, "content", None) or (r.get("content") or r.get("snippet") if isinstance(r, dict) else "")
        lines = [
            f"[{idx}] {title}",
            f"URL: {url}",
        ]
        if content:
            lines.append(f"Snippet: {content}")
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)


