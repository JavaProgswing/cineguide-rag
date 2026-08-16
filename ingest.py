"""Build a local CineGuide index from the bundled demo or TMDB."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rag import HashingEmbedder, Movie, SentenceTransformerEmbedder
from settings import settings

TMDB_BASE_URL = "https://api.themoviedb.org/3"
DEMO_PATH = Path(__file__).parent / "data" / "demo_movies.json"


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_tmdb_movies(api_key: str, pages: int, max_movies: int) -> list[Movie]:
    session = build_session()
    movies: list[Movie] = []
    seen: set[int] = set()
    for page in range(1, pages + 1):
        response = session.get(
            f"{TMDB_BASE_URL}/discover/movie",
            params={
                "api_key": api_key,
                "page": page,
                "sort_by": "popularity.desc",
                "include_adult": "false",
                "vote_count.gte": 100,
            },
            timeout=30,
        )
        response.raise_for_status()
        for item in response.json()["results"]:
            movie_id = int(item["id"])
            if movie_id in seen or not item.get("overview"):
                continue
            detail_response = session.get(
                f"{TMDB_BASE_URL}/movie/{movie_id}",
                params={"api_key": api_key, "append_to_response": "keywords,credits"},
                timeout=30,
            )
            detail_response.raise_for_status()
            detail = detail_response.json()
            crew = detail.get("credits", {}).get("crew", [])
            director = next(
                (person["name"] for person in crew if person.get("job") == "Director"), ""
            )
            movies.append(
                Movie(
                    id=movie_id,
                    title=detail.get("title", "Untitled"),
                    overview=detail.get("overview", ""),
                    genres=[genre["name"] for genre in detail.get("genres", [])],
                    release_date=detail.get("release_date", ""),
                    rating=float(detail.get("vote_average", 0.0)),
                    runtime=detail.get("runtime"),
                    keywords=[
                        keyword["name"]
                        for keyword in detail.get("keywords", {}).get("keywords", [])
                    ][:15],
                    cast=[person["name"] for person in detail.get("credits", {}).get("cast", [])][
                        :8
                    ],
                    director=director,
                    poster_path=detail.get("poster_path"),
                )
            )
            seen.add(movie_id)
            if len(movies) >= max_movies:
                return movies
            time.sleep(0.04)
    return movies


def load_demo_movies() -> list[Movie]:
    with DEMO_PATH.open("r", encoding="utf-8") as handle:
        return [Movie(**item) for item in json.load(handle)]


def write_index(movies: list[Movie], embedding_backend: str) -> None:
    if not movies:
        raise ValueError("No movies were available to index")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    embedder = (
        HashingEmbedder()
        if embedding_backend == "hashing"
        else SentenceTransformerEmbedder(settings.embedding_model)
    )
    embeddings = embedder.encode([movie.document_text() for movie in movies])
    with settings.documents_path.open("w", encoding="utf-8") as handle:
        for movie in movies:
            handle.write(json.dumps(movie.to_dict(), ensure_ascii=False) + "\n")
    np.save(settings.embeddings_path, np.asarray(embeddings, dtype=np.float32))
    settings.index_metadata_path.write_text(
        json.dumps(
            {
                "movie_count": len(movies),
                "embedding_backend": embedding_backend,
                "embedding_model": settings.embedding_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("demo", "tmdb"), default="demo")
    parser.add_argument("--pages", type=int, default=25, help="TMDB discovery pages")
    parser.add_argument("--max-movies", type=int, default=500)
    parser.add_argument(
        "--embedding-backend",
        choices=("sentence-transformers", "hashing"),
        default="sentence-transformers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "tmdb":
        if not settings.tmdb_api_key:
            raise SystemExit("TMDB_API_KEY is required for --source tmdb")
        movies = fetch_tmdb_movies(settings.tmdb_api_key, args.pages, args.max_movies)
    else:
        movies = load_demo_movies()
    write_index(movies, args.embedding_backend)
    print(f"Indexed {len(movies)} movies in {settings.data_dir.resolve()}")


if __name__ == "__main__":
    main()
