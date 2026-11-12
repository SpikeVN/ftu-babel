"""Database connection pooling and SQLite management for Babel."""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


class SQLiteConnectionPool:
    """Thread-safe persistent connection pool for SQLite with sqlite-vec and performance PRAGMAs."""

    def __init__(self, db_path: str, max_connections: int = 8):
        self.db_path = db_path
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=max_connections)
        self._max_connections = max_connections
        self._created = 0
        self._lock = threading.Lock()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=60.0,
            check_same_thread=False,
        )
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as e:
            logger.warning("Could not load sqlite_vec extension: %s", e)

        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA mmap_size = 2147483648;")
        conn.execute("PRAGMA cache_size = -64000;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        return conn

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = None
        try:
            conn = self._pool.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._created < self._max_connections:
                    conn = self._create_connection()
                    self._created += 1
            if conn is None:
                conn = self._pool.get(timeout=30.0)
        try:
            yield conn
        finally:
            self._pool.put(conn)

    def close(self) -> None:
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass
