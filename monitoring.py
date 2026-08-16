"""SQLite request logging, feedback, and aggregate monitoring queries."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from rag import SearchResult
from settings import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    retrieval_method TEXT NOT NULL,
    retrieved_documents TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    feedback INTEGER CHECK (feedback IN (-1, 1))
);
"""


@contextmanager
def connect(path: Path = settings.database_path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(SCHEMA)
        yield connection
        connection.commit()
    finally:
        connection.close()


def log_request(
    query: str,
    answer: str,
    latency_ms: float,
    retrieval_method: str,
    results: Sequence[SearchResult],
    token_usage: dict[str, int],
    path: Path = settings.database_path,
) -> int:
    documents = [
        {"id": result.movie.id, "title": result.movie.title, "rank": result.rank}
        for result in results
    ]
    with connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO requests (
                created_at, query, answer, latency_ms, retrieval_method,
                retrieved_documents, prompt_tokens, completion_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                query,
                answer,
                latency_ms,
                retrieval_method,
                json.dumps(documents),
                int(token_usage.get("prompt_tokens", 0)),
                int(token_usage.get("completion_tokens", 0)),
            ),
        )
        return int(cursor.lastrowid)


def save_feedback(request_id: int, value: int, path: Path = settings.database_path) -> None:
    if value not in (-1, 1):
        raise ValueError("Feedback must be 1 or -1")
    with connect(path) as connection:
        cursor = connection.execute(
            "UPDATE requests SET feedback = ? WHERE id = ?", (value, request_id)
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Unknown request id: {request_id}")


def request_rows(path: Path = settings.database_path) -> list[dict]:
    with connect(path) as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, query, latency_ms, retrieval_method,
                   prompt_tokens, completion_tokens, feedback
            FROM requests ORDER BY id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def summary(path: Path = settings.database_path) -> dict[str, float | int]:
    with connect(path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total_queries,
                   COALESCE(AVG(latency_ms), 0) AS average_latency_ms,
                   COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS total_tokens,
                   COALESCE(SUM(feedback = 1), 0) AS positive_feedback,
                   COALESCE(SUM(feedback = -1), 0) AS negative_feedback
            FROM requests
            """
        ).fetchone()
    return dict(row)
