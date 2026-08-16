"""Compare keyword, vector, and hybrid retrieval with Hit Rate and MRR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag import MovieIndex

QUESTIONS_PATH = Path(__file__).with_name("questions.json")
RESULTS_DIR = Path(__file__).with_name("results")


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    return next(
        (1.0 / rank for rank, movie_id in enumerate(retrieved, 1) if movie_id in relevant), 0.0
    )


def evaluate_method(index: MovieIndex, questions: list[dict], method: str, top_k: int) -> dict:
    hits = 0
    reciprocal_ranks: list[float] = []
    details: list[dict] = []
    for item in questions:
        results = index.search(item["query"], method=method, top_k=top_k)
        retrieved = [result.movie.id for result in results]
        relevant = set(item["relevant_ids"])
        hit = bool(relevant.intersection(retrieved))
        rr = reciprocal_rank(retrieved, relevant)
        hits += int(hit)
        reciprocal_ranks.append(rr)
        details.append({**item, "retrieved_ids": retrieved, "hit": hit, "reciprocal_rank": rr})
    count = len(questions)
    return {
        "method": method,
        "top_k": top_k,
        "question_count": count,
        "hit_rate": hits / count,
        "mrr": sum(reciprocal_ranks) / count,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "retrieval.json")
    args = parser.parse_args()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    index = MovieIndex.load()
    evaluations = [
        evaluate_method(index, questions, method, args.top_k)
        for method in ("keyword", "vector", "hybrid")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
    print(f"{'Method':<12} {'Hit Rate':>10} {'MRR':>10}")
    for result in evaluations:
        print(f"{result['method']:<12} {result['hit_rate']:>10.3f} {result['mrr']:>10.3f}")
    print(f"\nDetailed results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
