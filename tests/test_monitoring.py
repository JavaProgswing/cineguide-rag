from __future__ import annotations

from pathlib import Path

from monitoring import log_request, request_rows, save_feedback, summary
from rag import Movie, SearchResult


def test_request_and_feedback_round_trip(tmp_path: Path) -> None:
    database = tmp_path / "test.db"
    movie = Movie(id=42, title="Test Movie", overview="A test.", genres=["Drama"])
    result = SearchResult(movie=movie, score=0.5, method="hybrid", rank=1)
    request_id = log_request(
        "test query",
        "test answer",
        123.4,
        "hybrid",
        [result],
        {"prompt_tokens": 10, "completion_tokens": 5},
        path=database,
    )
    save_feedback(request_id, 1, path=database)

    rows = request_rows(path=database)
    totals = summary(path=database)
    assert rows[0]["feedback"] == 1
    assert totals == {
        "total_queries": 1,
        "average_latency_ms": 123.4,
        "total_tokens": 15,
        "positive_feedback": 1,
        "negative_feedback": 0,
    }
