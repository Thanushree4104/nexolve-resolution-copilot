import json
import re
import time
from pathlib import Path

from app.services.retriever import HybridRetriever
from app.services.rag_answer import RAGAnswerService
from app.llm.factory import get_provider


EVAL_FILE = Path("data/eval/retrieval_eval.jsonl")
OUTPUT_FILE = Path("data/eval/answer_results.jsonl")

EVAL_LIMIT = 5


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def classify_error(error):
    if not error:
        return "success"

    error_lower = error.lower()

    if (
        "rate limit" in error_lower
        or "rate_limit" in error_lower
        or "429" in error_lower
    ):
        return "llm_rate_limit"

    if "guardrail" in error_lower:
        return "guardrail_failure"

    if "citation validation" in error_lower:
        return "citation_failure"

    if "retrieval" in error_lower:
        return "retrieval_failure"

    return "other_error"


def citation_count(answer):
    if not answer:
        return 0

    return len(
        re.findall(
            r"\[KB-[A-Za-z0-9_-]+\]",
            answer,
        )
    )


def answer_word_count(answer):
    if not answer:
        return 0

    return len(
        re.findall(
            r"\b\w+\b",
            answer,
            flags=re.UNICODE,
        )
    )


def hit_at_k(retrieved_ids, relevant_ids, k):
    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    return bool(retrieved & relevant)


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)

    for rank, kb_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if kb_id in relevant:
            return round(1.0 / rank, 4)

    return 0.0


def main():
    evaluation_data = load_jsonl(
        EVAL_FILE
    )[:EVAL_LIMIT]

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

    for index, item in enumerate(
        evaluation_data,
        start=1,
    ):
        ticket_id = item["ticket_id"]
        query = item["query"]

        relevant_ids = set(
            item["relevant_ids"]
        )

        print(
            f"[{index}/{total}] {ticket_id}"
        )

        start = time.perf_counter()

        answer = ""
        error = None

        try:
            answer = service.answer(query)

        except Exception as exc:
            error = str(exc)

            print(
                f"  ERROR: {error}"
            )

        latency_ms = round(
            (
                time.perf_counter()
                - start
            )
            * 1000,
            1,
        )

        # Retrieve independently so that
        # retrieval quality can be evaluated
        # even when answer generation fails.
        try:
            retrieved = retriever.search(
                query,
                top_k=3,
            )

            retrieved_ids = [
                article["id"]
                for article in retrieved
            ]

        except Exception as exc:
            retrieved_ids = []

            if error is None:
                error = (
                    "retrieval_error: "
                    + str(exc)
                )

        status = classify_error(error)

        r1 = hit_at_k(
            retrieved_ids,
            relevant_ids,
            1,
        )

        r3 = hit_at_k(
            retrieved_ids,
            relevant_ids,
            3,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        citations = citation_count(
            answer
        )

        words = answer_word_count(
            answer
        )

        results.append(
            {
                "ticket_id": ticket_id,
                "query": query,
                "relevant_ids": sorted(
                    relevant_ids
                ),
                "retrieved_ids": retrieved_ids,

                # Retrieval diagnostics
                "retrieval_hit@1": r1,
                "retrieval_hit@3": r3,
                "retrieval_rr": rr,

                # Answer diagnostics
                "answer": answer,
                "answer_available": bool(
                    answer.strip()
                ),
                "answer_word_count": words,
                "citation_count": citations,

                # Execution diagnostics
                "error": error,
                "status": status,
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

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    successful = sum(
        result["status"] == "success"
        for result in results
    )

    answers_available = sum(
        result["answer_available"]
        for result in results
    )

    guardrail_failures = sum(
        result["status"]
        == "guardrail_failure"
        for result in results
    )

    citation_failures = sum(
        result["status"]
        == "citation_failure"
        for result in results
    )

    rate_limits = sum(
        result["status"]
        == "llm_rate_limit"
        for result in results
    )

    retrieval_failures = sum(
        result["status"]
        == "retrieval_failure"
        for result in results
    )

    other_errors = sum(
        result["status"]
        == "other_error"
        for result in results
    )

    retrieval_hit1 = sum(
        result["retrieval_hit@1"]
        for result in results
    )

    retrieval_hit3 = sum(
        result["retrieval_hit@3"]
        for result in results
    )

    mean_rr = (
        sum(
            result["retrieval_rr"]
            for result in results
        )
        / len(results)
        if results
        else 0.0
    )

    generated_results = [
        result
        for result in results
        if result["answer_available"]
    ]

    mean_answer_words = (
        sum(
            result["answer_word_count"]
            for result in generated_results
        )
        / len(generated_results)
        if generated_results
        else 0.0
    )

    citation_presence = (
        sum(
            result["citation_count"] > 0
            for result in generated_results
        )
        / len(generated_results)
        if generated_results
        else 0.0
    )

    print()
    print("=" * 70)
    print("ANSWER EVALUATION")
    print("=" * 70)

    print(
        f"Queries attempted       : {len(results)}"
    )

    print(
        f"Answers generated       : {answers_available}"
    )

    print(
        f"Successful pipeline     : {successful}"
    )

    print(
        f"Guardrail failures      : {guardrail_failures}"
    )

    print(
        f"Citation failures       : {citation_failures}"
    )

    print(
        f"LLM rate-limit failures : {rate_limits}"
    )

    print(
        f"Retrieval failures      : {retrieval_failures}"
    )

    print(
        f"Other errors            : {other_errors}"
    )

    print("-" * 70)

    print(
        f"Retrieval Hit@1         : "
        f"{retrieval_hit1 / len(results):.4f}"
        if results
        else "Retrieval Hit@1         : 0.0000"
    )

    print(
        f"Retrieval Hit@3         : "
        f"{retrieval_hit3 / len(results):.4f}"
        if results
        else "Retrieval Hit@3         : 0.0000"
    )

    print(
        f"Retrieval MRR           : "
        f"{mean_rr:.4f}"
    )

    print(
        f"Mean answer words       : "
        f"{mean_answer_words:.1f}"
    )

    print(
        f"Citation presence rate  : "
        f"{citation_presence:.4f}"
    )

    print("=" * 70)

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()