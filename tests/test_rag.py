from __future__ import annotations

import numpy as np

from rag import HashingEmbedder, Movie, MovieIndex, fallback_answer, parse_query_filters


def sample_index() -> MovieIndex:
    movies = [
        Movie(
            id=1,
            title="Red Planet",
            overview="Astronauts use science to survive on Mars.",
            genres=["Science Fiction", "Drama"],
            release_date="2018-01-01",
            runtime=110,
            keywords=["mars", "survival"],
        ),
        Movie(
            id=2,
            title="Haunted Station",
            overview="A space crew encounters a terrifying presence.",
            genres=["Science Fiction", "Horror"],
            release_date="2020-01-01",
            runtime=95,
            keywords=["space", "monster"],
        ),
        Movie(
            id=3,
            title="Robot Friends",
            overview="A family befriends a helpful robot.",
            genres=["Animation", "Family"],
            release_date="2014-01-01",
            runtime=90,
            keywords=["robot", "friendship"],
        ),
    ]
    embedder = HashingEmbedder(dimensions=64)
    embeddings = embedder.encode([movie.document_text() for movie in movies])
    return MovieIndex(movies, embeddings, embedder)


def test_query_filters_parse_constraints() -> None:
    filters = parse_query_filters("sci-fi after 2015 under 2 hours without horror")
    assert filters.after_year == 2015
    assert filters.max_runtime == 120
    assert "Horror" in filters.excluded_genres


def test_search_applies_constraints() -> None:
    results = sample_index().hybrid_search(
        "space survival after 2015 under 120 minutes without horror", top_k=3
    )
    assert [result.movie.id for result in results] == [1]


def test_all_retrievers_return_ranked_results() -> None:
    index = sample_index()
    for method in ("keyword", "vector", "hybrid"):
        results = index.search("family robot", method=method, top_k=2)
        assert results
        assert results[0].rank == 1
        assert all(np.isfinite(result.score) for result in results)


def test_fallback_answer_is_grounded_in_result_ids() -> None:
    results = sample_index().hybrid_search("Mars survival", top_k=1)
    answer = fallback_answer("Mars survival", results)
    assert results[0].movie.title in answer
    assert f"TMDB IDs): {results[0].movie.id}" in answer
