from __future__ import annotations

from evaluation.retrieval import reciprocal_rank


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([10, 20, 30], {20, 40}) == 0.5
    assert reciprocal_rank([10, 20, 30], {40}) == 0.0
