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

_SOCIAL_BAR_RE = re.compile(
    r"^(?:(?:behance|evernote|facebook|instagram|vkontakte|twitter|linkedin|youtube|pinterest)\s*){2,}$",
    flags=re.IGNORECASE,
)

_BAD_TITLE_RE = re.compile(
    r"^(?:[><•·\-_=*.\s]+|(?:[><•·\-_=*.\s]*)(?:xem chi tiết.*|thông tin chi tiết.*|chi tiết.*|tại đây.*|link\.?))$",
    flags=re.IGNORECASE,
)


def extract_title_from_text(text: str, url: str = "") -> str:
    """Extract a clean title from the first markdown heading, or fall back to the URL slug."""
    if text:
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#"):
                heading = clean_normal_text(s.lstrip("#").strip())
                if heading and not _BOILERPLATE_LINE_RE.match(heading) and not _BAD_TITLE_RE.match(heading):
                    return heading[:200]
    if url:
        decoded = urllib.parse.unquote(url).rstrip("/")
        slug = decoded.rsplit("/", 1)[-1]
        slug = re.sub(r"^\d+-", "", slug)
        slug = re.sub(r"[-_]+", " ", slug).strip()
        if slug:
            return slug[:200]
    return ""


def resolve_document_title(title: str, text: str = "", url: str = "") -> str:
    """Return a clean non-boilerplate document title, falling back to headings or URL slug."""
    cleaned = re.sub(r"\\([\\`*{}\[\]()#+\-.!_>])", r"\1", (title or "").strip())
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    if cleaned and not _BAD_TITLE_RE.match(cleaned) and not _BOILERPLATE_LINE_RE.match(cleaned):
        return cleaned[:200]
    return extract_title_from_text(text, url)


def clean_passage_lines(text: str) -> str:
    """Normalize text and strip CMS boilerplate/social-share lines without truncating."""
    cleaned = clean_normal_text(text)
    if not cleaned:
        return ""
    kept: list[str] = []
    for line in cleaned.splitlines():
        s = line.strip()
        if (
            not s
            or _BOILERPLATE_LINE_RE.match(s)
            or _SOCIAL_BAR_RE.match(s)
            or s in ("(", ")", "[", "]")
        ):
            continue
        kept.append(s)
    return "\n".join(kept).strip() or cleaned


def clean_passage_snippet(text: str, max_chars: int = 900) -> str:
    """Clean passage text and strip common CMS boilerplate lines before truncating."""
    joined = clean_passage_lines(text)
    if not joined:
        return ""
    if len(joined) > max_chars:
        return joined[:max_chars].rstrip() + "..."
    return joined


def _truncate_cleanly(text: str, max_chars: int) -> str:
    """Truncate text to at most max_chars at a word/newline boundary with trailing ellipsis."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_ws = max(cut.rfind(" "), cut.rfind("\n"))
    if last_ws > max_chars // 2:
        cut = cut[:last_ws]
    return cut.rstrip() + "..."


def extract_centered_passage(parent_text: str, anchor_text: str, max_chars: int = 1200) -> str:
    """
    Extract a window of at most `max_chars` from `parent_text` centered on `anchor_text`,
    so mid/tail child-chunk evidence is never lost to head truncation.
    """
    parent_clean = clean_passage_lines(parent_text)
    anchor_clean = clean_passage_lines(anchor_text)

    if not parent_clean:
        return _truncate_cleanly(anchor_clean, max_chars) if anchor_clean else ""
    if len(parent_clean) <= max_chars:
        return parent_clean

    span: tuple[int, int] | None = None
    if anchor_clean:
        pos = parent_clean.find(anchor_clean)
        if pos != -1:
            span = (pos, pos + len(anchor_clean))
        else:
            parent_lower = parent_clean.lower()
            anchor_lower = anchor_clean.lower()
            pos = parent_lower.find(anchor_lower)
            if pos != -1:
                span = (pos, pos + len(anchor_lower))
            else:
                core = anchor_clean.strip(". \n")
                core_lower = core.lower()
                probes: list[str] = []
                if len(core_lower) >= 4:
                    probes.append(core_lower[:80].strip())
                    if len(core_lower) > 80:
                        mid = max(0, (len(core_lower) - 80) // 2)
                        probes.append(core_lower[mid : mid + 80].strip())
                for seg in sorted((s.strip() for s in core_lower.split("...")), key=len, reverse=True):
                    if len(seg) >= 12:
                        probes.append(seg[:80].strip())
                for probe in probes:
                    if not probe:
                        continue
                    p = parent_lower.find(probe)
                    if p != -1:
                        span = (p, min(len(parent_clean), p + max(len(probe), min(len(core_lower), 400))))
                        break

        if span is None:
            combined = f"{anchor_clean}\n...\n{parent_clean}"
            return _truncate_cleanly(combined, max_chars)
    else:
        return _truncate_cleanly(parent_clean, max_chars)

    start, end = span
    match_len = end - start
    if match_len >= max_chars:
        win_start = start
        win_end = min(len(parent_clean), start + max_chars)
    else:
        pad = (max_chars - match_len) // 2
        win_start = max(0, start - pad)
        win_end = win_start + max_chars
        if win_end > len(parent_clean):
            win_end = len(parent_clean)
            win_start = max(0, win_end - max_chars)

    # Snap win_start forward up to 40 chars to a space/newline boundary (never past start)
    if win_start > 0 and not parent_clean[win_start - 1].isspace():
        search_limit = min(start, win_start + 40)
        ws_pos = -1
        for i in range(win_start, search_limit):
            if parent_clean[i] in (" ", "\n"):
                ws_pos = i
                break
        if ws_pos != -1:
            win_start = min(start, ws_pos + 1)

    # Snap win_end backward up to 40 chars to a space/newline boundary (never before end)
    if win_end < len(parent_clean) and not parent_clean[win_end].isspace():
        search_floor = max(end, win_end - 40)
        ws_pos = max(
            parent_clean.rfind(" ", search_floor, win_end),
            parent_clean.rfind("\n", search_floor, win_end),
        )
        if ws_pos >= end:
            win_end = ws_pos

    excerpt = parent_clean[win_start:win_end].strip()
    if win_start > 0:
        excerpt = "..." + excerpt
    if win_end < len(parent_clean):
        excerpt = excerpt + "..."
    return excerpt


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Safely extract a field from a dict or model attribute without treating 0 as falsy."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def format_search_results_as_text(results: Sequence[Any]) -> str:
    """Format search results into clean, token-efficient plain text for RAG LLM ingestion."""
    if not results:
        return "No results found."

    blocks = []
    for idx, r in enumerate(results, 1):
        doc_id = _get_field(r, "doc_id", "")
        raw_title = _get_field(r, "title", "") or ""
        url = _get_field(r, "url", "") or ""
        score = _get_field(r, "score", 0.0)
        dt = _get_field(r, "date", "") or ""
        summary = _get_field(r, "summary", "") or ""
        content = _get_field(r, "content", None)
        if content is None and isinstance(r, dict):
            content = r.get("snippet", "")
        content_str = str(content or "").strip()
        chunk_index = _get_field(r, "chunk_index", None)
        total_chunks = _get_field(r, "total_chunks", None)

        title = resolve_document_title(str(raw_title), content_str, str(url))
        score_val = float(score) if score is not None else 0.0
        cite_label = (title or extract_title_from_text("", str(url)) or str(url)).replace("[", "(").replace("]", ")")

        if chunk_index is not None and total_chunks is not None and int(total_chunks) > 0:
            header = f"[{idx}] Document ID: {doc_id} (Score: {score_val:.4f}) | Chunk: {chunk_index} of 0..{int(total_chunks) - 1}"
        else:
            header = f"[{idx}] Document ID: {doc_id} (Score: {score_val:.4f})"

        lines = [header]
        if title:
            lines.append(f"Title: {title}")
        lines.append(f"URL: {url}")
        if url:
            lines.append(f"Cite: [{cite_label}]({url})")
        if dt:
            lines.append(f"Date: {dt}")
        if summary:
            lines.append(f"Summary (auto-generated): {summary}")
        if content_str:
            lines.append(f"Matched Passage:\n{content_str}")
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)


def format_documents_as_text(documents: Sequence[dict[str, Any]]) -> str:
    """Format full or windowed document details into clean plain text for RAG ingestion."""
    if not documents:
        return "No documents found."

    blocks = []
    for doc in documents:
        url = doc.get("url", "") or ""
        content = (doc.get("content", "") or "").strip()
        title = resolve_document_title(doc.get("title", "") or "", content, url)
        lines = [
            f"Title: {title}",
            f"Document ID: {doc.get('doc_id', '')}",
            f"URL: {url}",
        ]
        if doc.get("date"):
            lines.append(f"Date: {doc['date']}")
        total_chunks = doc.get("total_chunks")
        if total_chunks is not None and int(total_chunks) > 0:
            tc = int(total_chunks)
            if doc.get("chunk_start") is not None and doc.get("chunk_end") is not None:
                lines.append(f"Sections: chunks {doc['chunk_start']}..{doc['chunk_end']} of 0..{tc - 1}")
            else:
                lines.append(f"Total Chunks: {tc} (0..{tc - 1})")
        if doc.get("summary"):
            lines.append(f"Summary: {doc['summary']}")
        header_text = "\n".join(lines)
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


