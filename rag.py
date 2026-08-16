"""Hybrid retrieval and grounded answer generation for CineGuide."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from settings import Settings, settings

TOKEN_RE = re.compile(r"[a-z0-9]+")
KNOWN_GENRES = {
    "action",
    "adventure",
    "animation",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "family",
    "fantasy",
    "history",
    "horror",
    "music",
    "mystery",
    "romance",
    "science fiction",
    "sci-fi",
    "thriller",
    "war",
    "western",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("sci-fi", "science fiction"))


@dataclass(frozen=True)
class Movie:
    id: int
    title: str
    overview: str
    genres: list[str]
    release_date: str = ""
    rating: float = 0.0
    runtime: int | None = None
    keywords: list[str] = field(default_factory=list)
    cast: list[str] = field(default_factory=list)
    director: str = ""
    poster_path: str | None = None

    @property
    def year(self) -> int | None:
        try:
            return int(self.release_date[:4])
        except (TypeError, ValueError):
            return None

    def document_text(self) -> str:
        return "\n".join(
            (
                f"Title: {self.title}",
                f"Overview: {self.overview}",
                f"Genres: {', '.join(self.genres)}",
                f"Release year: {self.year or 'unknown'}",
                f"Runtime: {self.runtime or 'unknown'} minutes",
                f"Rating: {self.rating}",
                f"Director: {self.director or 'unknown'}",
                f"Cast: {', '.join(self.cast)}",
                f"Keywords: {', '.join(self.keywords)}",
            )
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    movie: Movie
    score: float
    method: str
    rank: int


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Lazy-loading sentence-transformer adapter."""

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._load().encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Small deterministic fallback used by tests and constrained environments."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        output = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                output[row, hash(token) % self.dimensions] += 1.0
            norm = np.linalg.norm(output[row])
            if norm:
                output[row] /= norm
        return output


@dataclass(frozen=True)
class QueryFilters:
    after_year: int | None = None
    before_year: int | None = None
    max_runtime: int | None = None
    excluded_genres: tuple[str, ...] = ()


def parse_query_filters(query: str) -> QueryFilters:
    lowered = query.lower()
    after = re.search(r"(?:after|since)\s+(19\d{2}|20\d{2})", lowered)
    before = re.search(r"before\s+(19\d{2}|20\d{2})", lowered)
    runtime = re.search(r"(?:under|less than|max(?:imum)?)\s+(\d+)\s*(?:minutes?|mins?)", lowered)
    hour_runtime = re.search(
        r"(?:under|less than|max(?:imum)?)\s+(\d+(?:\.\d+)?)\s*hours?", lowered
    )
    excluded: list[str] = []
    for genre in KNOWN_GENRES:
        pattern = rf"(?:no|not|without|nothing)\s+(?:any\s+)?{re.escape(genre)}"
        if re.search(pattern, lowered):
            excluded.append("Science Fiction" if genre == "sci-fi" else genre.title())
    max_runtime = int(runtime.group(1)) if runtime else None
    if hour_runtime:
        max_runtime = int(float(hour_runtime.group(1)) * 60)
    return QueryFilters(
        after_year=int(after.group(1)) if after else None,
        before_year=int(before.group(1)) if before else None,
        max_runtime=max_runtime,
        excluded_genres=tuple(excluded),
    )


class MovieIndex:
    def __init__(
        self,
        movies: Sequence[Movie],
        embeddings: np.ndarray,
        embedder: Embedder,
    ) -> None:
        if len(movies) != len(embeddings):
            raise ValueError("Movie and embedding counts do not match")
        self.movies = list(movies)
        self.embeddings = self._normalize(np.asarray(embeddings, dtype=np.float32))
        self.embedder = embedder
        self._documents = [movie.document_text() for movie in self.movies]
        self._bm25 = BM25Okapi([tokenize(document) for document in self._documents])

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    @classmethod
    def load(
        cls,
        config: Settings = settings,
        embedder: Embedder | None = None,
    ) -> MovieIndex:
        if not config.documents_path.exists() or not config.embeddings_path.exists():
            raise FileNotFoundError(
                "Search index is missing. Run `python ingest.py --source demo` first."
            )
        with config.documents_path.open("r", encoding="utf-8") as handle:
            movies = [Movie(**json.loads(line)) for line in handle if line.strip()]
        embeddings = np.load(config.embeddings_path)
        if embedder is None and config.index_metadata_path.exists():
            metadata = json.loads(config.index_metadata_path.read_text(encoding="utf-8"))
            if metadata.get("embedding_backend") == "hashing":
                embedder = HashingEmbedder(dimensions=int(embeddings.shape[1]))
        return cls(
            movies,
            embeddings,
            embedder or SentenceTransformerEmbedder(config.embedding_model),
        )

    def _allowed(self, movie: Movie, filters: QueryFilters) -> bool:
        genres = {genre.lower() for genre in movie.genres}
        if filters.after_year and (movie.year is None or movie.year <= filters.after_year):
            return False
        if filters.before_year and (movie.year is None or movie.year >= filters.before_year):
            return False
        if filters.max_runtime and (movie.runtime is None or movie.runtime > filters.max_runtime):
            return False
        return not any(genre.lower() in genres for genre in filters.excluded_genres)

    def _ranked_indices(self, scores: np.ndarray, query: str, pool: int) -> list[int]:
        filters = parse_query_filters(query)
        ordered = np.argsort(-scores)
        return [index for index in ordered if self._allowed(self.movies[index], filters)][:pool]

    def keyword_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
        indices = self._ranked_indices(scores, query, top_k)
        return [
            SearchResult(self.movies[index], float(scores[index]), "keyword", rank)
            for rank, index in enumerate(indices, 1)
        ]

    def vector_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vector = self._normalize(self.embedder.encode([query]))[0]
        scores = self.embeddings @ query_vector
        indices = self._ranked_indices(scores, query, top_k)
        return [
            SearchResult(self.movies[index], float(scores[index]), "vector", rank)
            for rank, index in enumerate(indices, 1)
        ]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.55,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        pool = min(max(top_k * 5, 20), len(self.movies))
        keyword = self.keyword_search(query, pool)
        vector = self.vector_search(query, pool)
        combined: dict[int, float] = {}
        movies: dict[int, Movie] = {}
        for result in keyword:
            movies[result.movie.id] = result.movie
            combined[result.movie.id] = combined.get(result.movie.id, 0.0) + (
                (1.0 - vector_weight) / (rrf_k + result.rank)
            )
        for result in vector:
            movies[result.movie.id] = result.movie
            combined[result.movie.id] = combined.get(result.movie.id, 0.0) + (
                vector_weight / (rrf_k + result.rank)
            )
        ordered = sorted(combined, key=combined.get, reverse=True)[:top_k]
        return [
            SearchResult(movies[movie_id], combined[movie_id], "hybrid", rank)
            for rank, movie_id in enumerate(ordered, 1)
        ]

    def search(self, query: str, method: str = "hybrid", top_k: int = 5) -> list[SearchResult]:
        methods = {
            "keyword": self.keyword_search,
            "vector": self.vector_search,
            "hybrid": self.hybrid_search,
        }
        try:
            return methods[method](query, top_k)
        except KeyError as exc:
            raise ValueError(f"Unknown retrieval method: {method}") from exc


PROMPTS = {
    "concise": """You are CineGuide, a careful movie discovery assistant.
Use only the supplied movie context. Recommend up to five matches, explain each
match in one sentence, and mention any requested constraint that a movie does
not fully satisfy. Never invent movie facts. End with a short Sources line that
lists the TMDB movie IDs you used.""",
    "structured": """You are CineGuide. Answer only from the supplied movie
records. First summarize the user's intent. Then give up to five numbered
recommendations. For each include title, year, match reason, runtime, and one
possible caveat. Do not claim facts absent from context. Finish with a Sources
line containing the exact TMDB IDs used.""",
}


def build_context(results: Sequence[SearchResult]) -> str:
    return "\n\n---\n\n".join(
        f"TMDB ID: {result.movie.id}\n{result.movie.document_text()}" for result in results
    )


def fallback_answer(query: str, results: Sequence[SearchResult]) -> str:
    if not results:
        return "I could not find a movie in the index that satisfies those constraints."
    lines = [f"Here are the strongest indexed matches for **{query}**:"]
    for number, result in enumerate(results, 1):
        movie = result.movie
        details = [str(movie.year)] if movie.year else []
        if movie.runtime:
            details.append(f"{movie.runtime} min")
        reason = movie.overview.rstrip(".")
        lines.append(f"{number}. **{movie.title}** ({', '.join(details)}) — {reason}.")
    sources = ", ".join(str(result.movie.id) for result in results)
    lines.append(f"\nSources (TMDB IDs): {sources}")
    return "\n".join(lines)


def generate_answer(
    query: str,
    results: Sequence[SearchResult],
    config: Settings = settings,
    prompt_name: str = "structured",
) -> tuple[str, dict[str, int]]:
    """Return a grounded answer and token usage; work without an API key."""
    if not config.llm_api_key:
        return fallback_answer(query, results), {"prompt_tokens": 0, "completion_tokens": 0}

    from openai import OpenAI

    client = OpenAI(api_key=config.llm_api_key, base_url=config.llm_base_url)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": PROMPTS[prompt_name]},
            {
                "role": "user",
                "content": f"USER QUERY:\n{query}\n\nRETRIEVED CONTEXT:\n{build_context(results)}",
            },
        ],
    )
    usage = response.usage
    return response.choices[0].message.content or "", {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }
