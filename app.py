"""Streamlit UI for CineGuide."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from monitoring import log_request, request_rows, save_feedback, summary
from rag import MovieIndex, generate_answer
from settings import settings

st.set_page_config(page_title="CineGuide", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading the movie index…")
def load_index() -> MovieIndex:
    return MovieIndex.load()


def show_movie_cards(results) -> None:
    st.subheader("Retrieved sources")
    columns = st.columns(min(5, len(results)))
    for column, result in zip(columns, results, strict=False):
        movie = result.movie
        with column:
            if movie.poster_path:
                st.image(f"https://image.tmdb.org/t/p/w342{movie.poster_path}")
            st.markdown(f"**{result.rank}. {movie.title}**")
            details = [str(movie.year)] if movie.year else []
            if movie.runtime:
                details.append(f"{movie.runtime} min")
            st.caption(" · ".join(details))
            st.caption(f"TMDB ID {movie.id} · score {result.score:.4f}")


def recommendations_page(index: MovieIndex) -> None:
    st.title("🎬 CineGuide")
    st.write("Describe the mood, subject, era, runtime, or people you want to watch.")
    with st.form("search_form"):
        query = st.text_input(
            "What do you want to watch?",
            placeholder="Cerebral sci-fi like Arrival, but no horror",
        )
        with st.expander("Advanced options"):
            method = st.selectbox("Retrieval method", ("vector", "hybrid", "keyword"))
            top_k = st.slider("Number of recommendations", 1, 8, 5)
        submitted = st.form_submit_button("Find movies", type="primary")

    if submitted and query.strip():
        started = time.perf_counter()
        with st.spinner("Searching the catalog…"):
            results = index.search(query.strip(), method=method, top_k=top_k)
            answer, tokens = generate_answer(query.strip(), results)
        latency_ms = (time.perf_counter() - started) * 1000
        request_id = log_request(query.strip(), answer, latency_ms, method, results, tokens)
        st.session_state["last_response"] = {
            "answer": answer,
            "results": results,
            "request_id": request_id,
            "latency_ms": latency_ms,
        }
        st.session_state.pop("feedback_saved", None)

    response = st.session_state.get("last_response")
    if not response:
        st.info(
            "Try: “family animation about robots” or “crime mystery after 2015 under 140 minutes”."
        )
        return
    st.markdown(response["answer"])
    st.caption(f"Completed in {response['latency_ms']:.0f} ms")
    show_movie_cards(response["results"])

    st.write("Was this recommendation useful?")
    positive, negative, message = st.columns((1, 1, 8))
    if positive.button("👍", disabled=bool(st.session_state.get("feedback_saved"))):
        save_feedback(response["request_id"], 1)
        st.session_state["feedback_saved"] = True
        st.rerun()
    if negative.button("👎", disabled=bool(st.session_state.get("feedback_saved"))):
        save_feedback(response["request_id"], -1)
        st.session_state["feedback_saved"] = True
        st.rerun()
    if st.session_state.get("feedback_saved"):
        message.success("Feedback saved. Thank you!")


def monitoring_page() -> None:
    st.title("📊 Monitoring")
    rows = request_rows()
    totals = summary()
    total_feedback = totals["positive_feedback"] + totals["negative_feedback"]
    positive_rate = 100 * totals["positive_feedback"] / total_feedback if total_feedback else 0
    metrics = st.columns(4)
    metrics[0].metric("Queries", totals["total_queries"])
    metrics[1].metric("Average latency", f"{totals['average_latency_ms']:.0f} ms")
    metrics[2].metric("Positive feedback", f"{positive_rate:.0f}%")
    metrics[3].metric("Tokens", totals["total_tokens"])
    if not rows:
        st.info("Monitoring charts appear after the first query.")
        return

    frame = pd.DataFrame(rows)
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    frame["day"] = frame["created_at"].dt.date
    frame["total_tokens"] = frame["prompt_tokens"] + frame["completion_tokens"]

    left, right = st.columns(2)
    with left:
        st.subheader("1. Queries per day")
        st.bar_chart(frame.groupby("day").size().rename("queries"))
        st.subheader("3. Retrieval methods")
        st.bar_chart(frame["retrieval_method"].value_counts().rename("queries"))
        st.subheader("5. Token usage over time")
        st.line_chart(frame.set_index("created_at")["total_tokens"])
    with right:
        st.subheader("2. Latency over time")
        st.line_chart(frame.set_index("created_at")["latency_ms"])
        st.subheader("4. Feedback distribution")
        labels = frame["feedback"].map({1.0: "positive", -1.0: "negative"}).fillna("none")
        st.bar_chart(labels.value_counts().rename("responses"))
        st.subheader("6. Most frequent queries")
        st.bar_chart(frame["query"].value_counts().head(10).rename("count"))


def main() -> None:
    st.sidebar.title("CineGuide")
    page = st.sidebar.radio("Navigate", ("Recommendations", "Monitoring"))
    try:
        index = load_index()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python ingest.py --source demo")
        st.stop()
    st.sidebar.caption(f"{len(index.movies)} movies indexed")
    st.sidebar.caption(
        f"LLM: {settings.llm_model}" if settings.llm_api_key else "LLM: grounded offline fallback"
    )
    if page == "Recommendations":
        recommendations_page(index)
    else:
        monitoring_page()


if __name__ == "__main__":
    main()
