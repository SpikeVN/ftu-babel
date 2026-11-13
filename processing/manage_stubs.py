"""
CLI management tool to isolate, export, inspect, and update document stubs in stash.db.

Usage:
  python processing/manage_stubs.py --summary
  python processing/manage_stubs.py --export pending_stubs.jsonl
  python processing/manage_stubs.py --export pending_pdfs.jsonl --type pdf
  python processing/manage_stubs.py --list-pending --type docx
  python processing/manage_stubs.py --update 10268 --text-file extracted_text.txt
  python processing/manage_stubs.py --mark-extracted 10268
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_connection(db_path: str = "stash.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def show_summary(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM document_stubs")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT status, count(*) FROM document_stubs GROUP BY 1 ORDER BY 2 DESC")
    status_counts = cursor.fetchall()

    cursor.execute("SELECT type, count(*) FROM document_stubs GROUP BY 1 ORDER BY 2 DESC")
    mime_counts = cursor.fetchall()

    cursor.execute("SELECT file_type, count(*) FROM document_stubs GROUP BY 1 ORDER BY 2 DESC")
    type_counts = cursor.fetchall()

    cursor.execute("SELECT source_domain, count(*) FROM document_stubs GROUP BY 1 ORDER BY 2 DESC LIMIT 10")
    domain_counts = cursor.fetchall()

    print(f"\n================ DOCUMENT STUBS SUMMARY ================")
    print(f"Total Stubs Tracked: {total}")
    print("\n--- By Status ---")
    for r in status_counts:
        print(f"  {r[0]:<15}: {r[1]:>5}")

    print("\n--- By Resolved MIME Type (type column) ---")
    for r in mime_counts:
        print(f"  {r[0] or 'unknown':<65}: {r[1]:>5}")

    print("\n--- By Original URL Stub Type ---")
    for r in type_counts:
        print(f"  {r[0]:<20}: {r[1]:>5}")

    print("\n--- Top Source Domains ---")
    for r in domain_counts:
        print(f"  {r[0]:<30}: {r[1]:>5}")
    print("========================================================\n")


def export_stubs(conn: sqlite3.Connection, output_file: str, file_type: str | None = None, status: str = "pending") -> None:
    cursor = conn.cursor()
    query = """
    SELECT doc_id, url, title, type, mime_type, file_type, source_domain, parent_doc_id,
           status, local_path, extracted_chars, file_created_at, file_modified_at, file_author, file_producer
    FROM document_stubs
    WHERE status = ?
    """
    params = [status]
    if file_type:
        query += " AND (type = ? OR file_type = ?)"
        params.extend([file_type, file_type])
    query += " ORDER BY doc_id ASC"

    rows = cursor.execute(query, params).fetchall()
    out_path = Path(output_file)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            record = dict(r)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Exported {len(rows)} stubs to {output_file}")


def list_pending(conn: sqlite3.Connection, file_type: str | None = None, limit: int = 50) -> None:
    cursor = conn.cursor()
    query = "SELECT doc_id, type, file_type, title, url FROM document_stubs WHERE status = 'pending'"
    params = []
    if file_type:
        query += " AND (type = ? OR file_type = ?)"
        params.extend([file_type, file_type])
    query += f" ORDER BY doc_id ASC LIMIT {limit}"

    rows = cursor.execute(query, params).fetchall()
    print(f"\nListing {len(rows)} pending stubs (limit={limit}):")
    for r in rows:
        print(f"  Doc #{r['doc_id']} [{r['type'] or r['file_type']}] \"{r['title']}\" -> {r['url'][:80]}")


def update_doc_text(conn: sqlite3.Connection, doc_id: int, text: str) -> None:
    cursor = conn.cursor()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Update documents table (FTS triggers update fts_documents automatically)
    cursor.execute("UPDATE documents SET markdown = ? WHERE id = ?", (text, doc_id))
    if cursor.rowcount == 0:
        print(f"Error: Document #{doc_id} not found in `documents` table.")
        return

    # Update document_stubs table
    cursor.execute("""
    UPDATE document_stubs
    SET status = 'extracted', extracted_chars = ?, status = 'extracted'
    WHERE doc_id = ?
    """, (len(text), doc_id))

    conn.commit()
    print(f"Successfully updated Doc #{doc_id} with {len(text)} characters of text and marked as 'extracted'.")


def main():
    parser = argparse.ArgumentParser(description="Manage and isolate document stubs in stash.db")
    parser.add_argument("--db", default="stash.db", help="Path to sqlite database (default: stash.db)")
    parser.add_argument("--summary", action="store_true", help="Show summary breakdown of all stubs")
    parser.add_argument("--export", metavar="FILE", help="Export pending stubs to JSONL file")
    parser.add_argument("--type", help="Filter by file_type (e.g. pdf, docx, doc, google_doc)")
    parser.add_argument("--status", default="pending", help="Filter by status for export (default: pending)")
    parser.add_argument("--list-pending", action="store_true", help="List pending stubs")
    parser.add_argument("--limit", type=int, default=50, help="Limit for list-pending (default: 50)")
    parser.add_argument("--update", type=int, metavar="DOC_ID", help="Doc ID to update with extracted text")
    parser.add_argument("--text-file", metavar="FILE", help="Path to plain text file to insert into document")
    parser.add_argument("--text", metavar="STR", help="Text string to insert into document")

    args = parser.parse_args()
    conn = get_connection(args.db)

    if args.summary:
        show_summary(conn)
    elif args.export:
        export_stubs(conn, args.export, file_type=args.type, status=args.status)
    elif args.list_pending:
        list_pending(conn, file_type=args.type, limit=args.limit)
    elif args.update:
        if args.text_file:
            content = Path(args.text_file).read_text(encoding="utf-8")
        elif args.text:
            content = args.text
        else:
            print("Error: Specify --text-file or --text when updating a doc.")
            sys.exit(1)
        update_doc_text(conn, args.update, content)
    else:
        show_summary(conn)

    conn.close()


if __name__ == "__main__":
    main()
