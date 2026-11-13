"""OpenWebUI Tool: FTU Babel Knowledge Base Retrieval (babel_query & babel_fetch).

Provides integration with the Babel hybrid knowledge-base search engine:
  - babel_query: Hybrid / semantic / keyword search with zerank-2 reranking.
  - babel_fetch: Full-document batch retrieval by document ID(s) in a single request.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

# Try importing requests if available in OpenWebUI environment
try:
    import requests
except ImportError:
    requests = None  # type: ignore

DEFAULT_API_URL = os.environ.get(
    "SEARCH_API_URL", "https://babel.falcolabs.org"
)
DEFAULT_API_KEY = os.environ.get("SEARCH_API_KEY", "")


class Tools:
    class Valves(BaseModel):
        SEARCH_API_URL: str = Field(
            default=DEFAULT_API_URL,
            description="Base URL for the Babel API service (e.g. https://babel.falcolabs.org)",
        )
        SEARCH_API_KEY: str = Field(
            default=DEFAULT_API_KEY,
            description="Optional API key for authorization (Bearer token)",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    def _config(self) -> "Valves":
        """Retrieve tool configuration valves safely."""
        valves = getattr(self, "valves", None)
        if valves is None:
            valves = self.Valves()
        return valves

    def _base_url(self) -> str:
        """Resolve the base URL by stripping trailing path suffixes like /api/search or /search."""
        url = self._config().SEARCH_API_URL.strip().rstrip("/")
        for suffix in ("/api/search", "/search", "/api/fetch", "/docs", "/api"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        return url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        """Construct standard HTTP headers."""
        headers = {
            "Accept": "text/plain, application/json",
            "Content-Type": "application/json",
            "User-Agent": "OpenWebUI-BabelTool/1.0",
        }
        key = self._config().SEARCH_API_KEY.strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _http_post(self, url: str, json_data: dict[str, Any], timeout: float = 60.0) -> tuple[int, str]:
        """Execute HTTP POST request with requests or fallback to urllib."""
        headers = self._headers()
        if requests is not None:
            resp = requests.post(url, json=json_data, headers=headers, timeout=timeout)
            return resp.status_code, resp.text

        data_bytes = json.dumps(json_data).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")

    def babel_query(
        self,
        query: str,
        mode: Literal["semantic", "keyword", "hybrid"] = "semantic",
        top_k: int = 10,
        format: Literal["text", "json"] = "text",
    ) -> str:
        """
        Search the FTU (Trường Đại học Ngoại thương) knowledge base.
        Returns relevant document passages with their doc_id, URL, date, summary, and score.

        Args:
            query: The search query string (use user's literal wording, typos included).
            mode: Search mode:
                  - 'semantic' (default): Pure vector search (hybrid_blend=1.0) for general topics, concepts, and questions.
                  - 'keyword': Pure BM25 keyword search (hybrid_blend=0.0) for person names, acronyms, codes, or course IDs
                    (e.g., KTQT301, FTDS, QLĐT).
                  - 'hybrid': 50/50 balanced blend of semantic vector and keyword search (hybrid_blend=0.5).
            top_k: Maximum number of passage results to return (1-100, default 10).
            format: Output format, either 'text' (default, citation-ready plain text for LLM) or 'json'.

        Returns:
            Citation-ready passages with doc_id, URL, publication date, summary, and clean text content.
        """
        base_url = self._base_url()
        endpoint = f"{base_url}/query"

        # Map search modes to server hybrid_blend
        # Server semantics: 1.0 = pure vector (semantic), 0.0 = pure keyword (BM25), 0.5 = hybrid
        mode_lower = (mode or "semantic").strip().lower()
        if mode_lower == "keyword":
            blend = 0.0
        elif mode_lower == "semantic":
            blend = 1.0
        else:
            blend = 0.5

        limit = max(1, min(int(top_k), 100))
        payload = {
            "query": query,
            "limit": limit,
            "hybrid_blend": blend,
            "format": format if format in ("text", "json") else "text",
        }

        try:
            status_code, body = self._http_post(endpoint, payload, timeout=60.0)
            if status_code == 404:
                # Fallback to legacy /search endpoint if /query not yet available
                status_code, body = self._http_post(f"{base_url}/search", payload, timeout=60.0)
            if status_code >= 400:
                return f"Error {status_code} querying Babel knowledge base: {body}"
            return body
        except Exception as exc:
            return f"Error querying Babel knowledge base: {exc}"

    def web_search(
        self,
        query: str,
        limit: int = 5,
        format: Literal["text", "json"] = "text",
    ) -> str:
        """
        Search the public web for real-time, external, or non-FTU information via SearXNG.
        Use as a fallback when FTU knowledge base (babel_query) does not contain the answer.

        Args:
            query: The web search query string.
            limit: Maximum number of search results to return (1-20, default 5).
            format: Output format, either 'text' (default) or 'json'.

        Returns:
            Web search result snippets with titles and URLs.
        """
        base_url = self._base_url()
        endpoint = f"{base_url}/web_search"
        payload = {
            "query": query,
            "limit": max(1, min(int(limit), 20)),
            "format": format if format in ("text", "json") else "text",
        }
        try:
            status_code, body = self._http_post(endpoint, payload, timeout=30.0)
            if status_code >= 400:
                return f"Error {status_code} performing web search: {body}"
            return body
        except Exception as exc:
            return f"Error performing web search: {exc}"

    def babel_fetch(
        self,
        doc_ids: Optional[Union[int, str, list[Union[int, str]]]] = None,
        doc_id: Optional[Union[int, str, list[Union[int, str]]]] = None,
        ids: Optional[Union[list[Union[int, str]], int, str]] = None,
        clean: bool = True,
        format: Literal["text", "json"] = "text",
    ) -> str:
        """
        Fetch the full document content and metadata for one or more doc_ids in a single batch request.
        Use when search passages in babel_query lack specific details, or when full text context is needed.

        Args:
            doc_ids: A document ID (e.g., 1525) or list of doc_ids (e.g., [1525, 1526]) returned by babel_query.
            doc_id: Optional alias for a single document ID or list of doc_ids.
            ids: Optional alias for doc_ids.
            clean: Whether to strip links, images, and headers so only normal text remains (default True).
            format: Output format, either 'text' (default, clean plain text) or 'json'.

        Returns:
            Full clean text content of the document(s) along with Title, URL, Date, and Summary.
        """
        # Collect and normalize all requested IDs
        target_ids: list[int] = []

        def _collect(raw: Any) -> None:
            if raw is None:
                return
            if isinstance(raw, int) and not isinstance(raw, bool):
                target_ids.append(raw)
                return
            if isinstance(raw, str):
                s = raw.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        parsed = json.loads(s)
                        _collect(parsed)
                        return
                    except Exception:
                        pass
                for part in s.split(","):
                    part = part.strip().rstrip("/")
                    if "/docs/" in part:
                        part = part.rsplit("/docs/", 1)[-1]
                    if part.isdigit():
                        target_ids.append(int(part))
                return
            if isinstance(raw, (list, tuple, set)):
                for item in raw:
                    _collect(item)

        for raw in (doc_ids, doc_id, ids):
            _collect(raw)

        # Deduplicate while preserving order
        target_ids = list(dict.fromkeys(target_ids))

        if not target_ids:
            return "Error: No valid numeric doc_id provided to babel_fetch."

        base_url = self._base_url()
        endpoint = f"{base_url}/docs/fetch"
        payload = {
            "doc_ids": target_ids,
            "clean": bool(clean),
            "format": format if format in ("text", "json") else "text",
        }

        try:
            status_code, body = self._http_post(endpoint, payload, timeout=60.0)
            if status_code >= 400:
                return f"Error {status_code} fetching documents from Babel: {body}"
            return body
        except Exception as exc:
            return f"Error fetching documents from Babel: {exc}"
