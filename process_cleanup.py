"""HTML & Markdown Cleanup Pipeline (backward-compatibility wrapper delegating to processing.cleanup)."""

import sys
from processing.cleanup import (
    HEADERS_TO_PURGE,
    clean_database,
    clean_document_record,
    clean_html,
    clean_markdown,
    create_html2text_converter,
    main,
    purge,
    purge_section,
    strip_header_nav,
    strip_header_nav_from_database,
)

__all__ = [
    "HEADERS_TO_PURGE",
    "purge_section",
    "clean_markdown",
    "strip_header_nav",
    "purge",
    "create_html2text_converter",
    "clean_html",
    "clean_document_record",
    "clean_database",
    "strip_header_nav_from_database",
    "main",
]

if __name__ == "__main__":
    main()
