import json
import time
from pathlib import Path

from app.services.retriever import HybridRetriever
from app.services.rag_answer import RAGAnswerService
from app.llm.factory import get_provider


EVAL_FILE = Path("data/eval/retrieval_eval.jsonl")
OUTPUT_FILE = Path("data/eval/answer_results.jsonl")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    evaluation_data = load_jsonl(EVAL_FILE)

    retriever = HybridRetriever()
    provider = get_provider()

    service = RAGAnswerService(
        provider=provider,
        retriever=retriever,
        max_articles=3,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    total = len(evaluation_data)

    for index, item in enumerate(evaluation_data, start=1):
        ticket_id = item["ticket_id"]
        query = item["query"]
        relevant_ids = set(item["relevant_ids"])

        print(
            f"[{index}/{total}] "
            f"{ticket_id}"
        )

        start = time.perf_counter()

        try:
            answer = service.answer(query)

            error = None

        except Exception as exc:
            answer = ""
            error = str(exc)

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            1,
        )

        # Retrieve independently so evaluation records
        # contain the actual retrieved KB IDs.
        try:
            retrieved = retriever.search(
                query,
                top_k=3,
            )

            retrieved_ids = [
                article["id"]
                for article in retrieved
            ]

        except Exception:
            retrieved_ids = []

        results.append(
            {
                "ticket_id": ticket_id,
                "query": query,
                "relevant_ids": sorted(relevant_ids),
                "retrieved_ids": retrieved_ids,
                "answer": answer,
                "error": error,
                "latency_ms": latency_ms,
            }
        )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        f"Answer evaluation completed."
    )
    print(
        f"Queries: {len(results)}"
    )
    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()