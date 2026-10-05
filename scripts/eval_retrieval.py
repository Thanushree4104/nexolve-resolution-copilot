import json
import math
import time
from pathlib import Path

import numpy as np

from app.services.retriever import HybridRetriever


EVAL_PATH = Path("data/eval/retrieval_eval.jsonl")
OUTPUT_PATH = Path("data/eval/retrieval_results.json")

TOP_K = 5


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def recall_at_k(
    ranked_ids,
    relevant_ids,
    k,
):
    relevant_ids = set(relevant_ids)

    return float(
        bool(
            set(ranked_ids[:k])
            & relevant_ids
        )
    )


def hit_at_k(
    ranked_ids,
    relevant_ids,
    k,
):
    return recall_at_k(
        ranked_ids,
        relevant_ids,
        k,
    )


def reciprocal_rank(
    ranked_ids,
    relevant_ids,
):
    relevant_ids = set(relevant_ids)

    for rank, kb_id in enumerate(
        ranked_ids,
        start=1,
    ):
        if kb_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    ranked_ids,
    relevant_ids,
    k,
):
    relevant_ids = set(relevant_ids)

    dcg = 0.0

    for rank, kb_id in enumerate(
        ranked_ids[:k],
        start=1,
    ):
        if kb_id in relevant_ids:
            dcg += 1.0 / math.log2(
                rank + 1
            )

    ideal_hits = min(
        len(relevant_ids),
        k,
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    return dcg / idcg


def evaluate_query(
    item,
    retrieved_ids,
):
    relevant_ids = item["relevant_ids"]

    return {
        "ticket_id": item["ticket_id"],
        "retrieved_ids": retrieved_ids,
        "relevant_ids": relevant_ids,

        "rr": reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        ),

        "recall@1": recall_at_k(
            retrieved_ids,
            relevant_ids,
            1,
        ),

        "recall@3": recall_at_k(
            retrieved_ids,
            relevant_ids,
            3,
        ),

        "recall@5": recall_at_k(
            retrieved_ids,
            relevant_ids,
            5,
        ),

        "hit@1": hit_at_k(
            retrieved_ids,
            relevant_ids,
            1,
        ),

        "hit@3": hit_at_k(
            retrieved_ids,
            relevant_ids,
            3,
        ),

        "hit@5": hit_at_k(
            retrieved_ids,
            relevant_ids,
            5,
        ),

        "ndcg@1": ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            1,
        ),

        "ndcg@3": ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            3,
        ),

        "ndcg@5": ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            5,
        ),
    }


def build_summary(results):
    if not results:
        return {
            "retriever": "hybrid",
            "queries": 0,
        }

    summary = {
        "retriever": "hybrid",
        "queries": len(results),

        "mrr": float(
            np.mean(
                [
                    result["rr"]
                    for result in results
                ]
            )
        ),
    }

    for k in (1, 3, 5):
        summary[f"recall@{k}"] = float(
            np.mean(
                [
                    result[
                        f"recall@{k}"
                    ]
                    for result in results
                ]
            )
        )

        summary[f"hit@{k}"] = float(
            np.mean(
                [
                    result[
                        f"hit@{k}"
                    ]
                    for result in results
                ]
            )
        )

        summary[f"ndcg@{k}"] = float(
            np.mean(
                [
                    result[
                        f"ndcg@{k}"
                    ]
                    for result in results
                ]
            )
        )

    return summary


def main():
    evaluation = load_jsonl(
        EVAL_PATH
    )

    print(
        f"Evaluation queries : "
        f"{len(evaluation)}"
    )

    print(
        "Retriever           : "
        "HybridRetriever "
        "(production path)"
    )

    print(
        f"Embedding model     : "
        f"intfloat/multilingual-e5-small"
    )

    print()

    print(
        "Initializing production "
        "HybridRetriever..."
    )

    retriever = HybridRetriever()

    print()

    results = []

    start = time.perf_counter()

    for position, item in enumerate(
        evaluation,
        start=1,
    ):
        query = item["query"]

        retrieval_start = (
            time.perf_counter()
        )

        retrieved = retriever.search(
            query,
            top_k=TOP_K,
        )

        retrieval_latency_ms = round(
            (
                time.perf_counter()
                - retrieval_start
            )
            * 1000,
            1,
        )

        retrieved_ids = [
            result["id"]
            for result in retrieved
        ]

        result = evaluate_query(
            item,
            retrieved_ids,
        )

        result[
            "retrieval_latency_ms"
        ] = retrieval_latency_ms

        results.append(result)

        if position % 50 == 0:
            print(
                f"Processed {position}/"
                f"{len(evaluation)} queries"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    summary = build_summary(
        results
    )

    print()

    print(
        f"Retrieval evaluation completed "
        f"in {elapsed:.2f}s"
    )

    print()

    print("=" * 90)
    print("RETRIEVAL EVALUATION")
    print("=" * 90)

    print(
        f"{'Retriever':<12}"
        f"{'R@1':>10}"
        f"{'R@3':>10}"
        f"{'R@5':>10}"
        f"{'MRR':>10}"
        f"{'nDCG@5':>12}"
    )

    print("-" * 90)

    print(
        f"{'hybrid':<12}"
        f"{summary['recall@1']:>10.4f}"
        f"{summary['recall@3']:>10.4f}"
        f"{summary['recall@5']:>10.4f}"
        f"{summary['mrr']:>10.4f}"
        f"{summary['ndcg@5']:>12.4f}"
    )

    print("=" * 90)

    output = {
        "evaluation_dataset": str(
            EVAL_PATH
        ),
        "retriever": (
            "app.services.retriever."
            "HybridRetriever"
        ),
        "retrieval_path": (
            "production"
        ),
        "top_k": TOP_K,
        "query_count": len(evaluation),
        "summary": summary,
        "per_query": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()

    print(
        f"Detailed results written to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()