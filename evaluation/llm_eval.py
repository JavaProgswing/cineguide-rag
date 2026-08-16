"""Compare two answer prompts with an LLM judge for relevance and faithfulness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

from rag import PROMPTS, MovieIndex, build_context, generate_answer
from settings import settings

QUESTIONS_PATH = Path(__file__).with_name("questions.json")
RESULTS_DIR = Path(__file__).with_name("results")
JUDGE_PROMPT = """You evaluate a movie RAG answer. Compare the answer only with
the user query and retrieved context. Return valid JSON with integer scores from
1 (poor) to 5 (excellent): {"relevance": N, "faithfulness": N, "clarity": N,
"reason": "short explanation"}. Faithfulness means every factual claim is
supported by the context. Do not include markdown fences."""


def judge(client: OpenAI, query: str, context: str, answer: str) -> dict:
    response = client.chat.completions.create(
        model=settings.llm_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"QUERY:\n{query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}",
            },
        ],
    )
    return json.loads(response.choices[0].message.content or "{}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "llm.json")
    args = parser.parse_args()
    if not settings.llm_api_key:
        raise SystemExit("LLM_API_KEY is required for LLM-as-judge evaluation")

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    index = MovieIndex.load()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))[: args.limit]
    rows: list[dict] = []
    for number, item in enumerate(questions, 1):
        results = index.hybrid_search(item["query"], top_k=5)
        context = build_context(results)
        for prompt_name in PROMPTS:
            answer, usage = generate_answer(item["query"], results, prompt_name=prompt_name)
            scores = judge(client, item["query"], context, answer)
            rows.append(
                {
                    "query": item["query"],
                    "prompt": prompt_name,
                    "answer": answer,
                    "usage": usage,
                    **scores,
                }
            )
        print(f"Evaluated {number}/{len(questions)}: {item['query']}")

    summary = {}
    for prompt_name in PROMPTS:
        prompt_rows = [row for row in rows if row["prompt"] == prompt_name]
        summary[prompt_name] = {
            metric: sum(float(row[metric]) for row in prompt_rows) / len(prompt_rows)
            for metric in ("relevance", "faithfulness", "clarity")
        }
    output = {"judge_model": settings.llm_model, "summary": summary, "details": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Detailed results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
