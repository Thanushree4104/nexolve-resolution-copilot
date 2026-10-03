import json
import math
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


EVAL_PATH = Path("data/eval/retrieval_eval.jsonl")
KB_PATH = Path("data/synthetic/kb_articles.jsonl")

MODEL = "intfloat/multilingual-e5-small"


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def tokenize(text):
    import re

    return re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE,
    )


def doc_text(article):
    title = article.get("title", "")
    class_id = article.get("class_id", "")
    product = article.get("product", "")
    root_cause = article.get("root_cause", "")

    symptoms = " ".join(
        article.get("symptoms", [])
    )

    questions = article.get(
        "diagnostic_questions",
        [],
    )

    diagnostic_questions = " ".join(
        q.get("q", "")
        for q in questions
    )

    diagnostic_guidance = " ".join(
        q.get("if_yes", "")
        + " "
        + q.get("if_no", "")
        for q in questions
    )

    return (
        f"Title: {title}. "
        f"Problem class: {class_id}. "
        f"Product: {product}. "
        f"Root cause: {root_cause}. "
        f"Symptoms: {symptoms}. "
        f"Diagnostic questions: {diagnostic_questions}. "
        f"Diagnostic guidance: {diagnostic_guidance}."
    )


def reciprocal_rank_fusion(
    dense_order,
    bm25_order,
    k=60,
):
    scores = {}

    for rank, index in enumerate(dense_order):
        scores[index] = (
            scores.get(index, 0.0)
            + 1.0 / (k + rank + 1)
        )

    for rank, index in enumerate(bm25_order):
        scores[index] = (
            scores.get(index, 0.0)
            + 1.0 / (k + rank + 1)
        )

    return sorted(
        scores,
        key=scores.get,
        reverse=True,
    )


def recall_at_k(ranked_ids, relevant_ids, k):
    return float(
        bool(
            set(ranked_ids[:k])
            & set(relevant_ids)
        )
    )


def hit_at_k(ranked_ids, relevant_ids, k):
    return recall_at_k(
        ranked_ids,
        relevant_ids,
        k,
    )


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant_ids = set(relevant_ids)

    for rank, kb_id in enumerate(
        ranked_ids,
        start=1,
    ):
        if kb_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k):
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


def evaluate(
    name,
    rankings,
    evaluation,
    ks=(1, 3, 5),
):
    results = []

    for item, ranked_indices in zip(
        evaluation,
        rankings,
    ):
        ranked_ids = [
            KB_IDS[index]
            for index in ranked_indices
        ]

        relevant_ids = item["relevant_ids"]

        results.append(
            {
                "ticket_id": item["ticket_id"],
                "retrieved_ids": ranked_ids,
                "relevant_ids": relevant_ids,
                "rr": reciprocal_rank(
                    ranked_ids,
                    relevant_ids,
                ),
                **{
                    f"recall@{k}": recall_at_k(
                        ranked_ids,
                        relevant_ids,
                        k,
                    )
                    for k in ks
                },
                **{
                    f"hit@{k}": hit_at_k(
                        ranked_ids,
                        relevant_ids,
                        k,
                    )
                    for k in ks
                },
                **{
                    f"ndcg@{k}": ndcg_at_k(
                        ranked_ids,
                        relevant_ids,
                        k,
                    )
                    for k in ks
                },
            }
        )

    summary = {
        "retriever": name,
        "queries": len(results),
        "mrr": float(
            np.mean(
                [r["rr"] for r in results]
            )
        ),
    }

    for k in ks:
        summary[f"recall@{k}"] = float(
            np.mean(
                [
                    r[f"recall@{k}"]
                    for r in results
                ]
            )
        )

        summary[f"hit@{k}"] = float(
            np.mean(
                [
                    r[f"hit@{k}"]
                    for r in results
                ]
            )
        )

        summary[f"ndcg@{k}"] = float(
            np.mean(
                [
                    r[f"ndcg@{k}"]
                    for r in results
                ]
            )
        )

    return summary, results


evaluation = load_jsonl(EVAL_PATH)
articles = load_jsonl(KB_PATH)

KB_IDS = [
    article["id"]
    for article in articles
]

DOCUMENTS = [
    doc_text(article)
    for article in articles
]

print(f"Evaluation queries : {len(evaluation)}")
print(f"KB articles        : {len(articles)}")
print(f"Embedding model    : {MODEL}")
print()

print("Loading embedding model...")
model = SentenceTransformer(MODEL)

print("Encoding KB...")
doc_vectors = model.encode(
    [
        "passage: " + text
        for text in DOCUMENTS
    ],
    normalize_embeddings=True,
    batch_size=32,
)

bm25 = BM25Okapi(
    [
        tokenize(text)
        for text in DOCUMENTS
    ]
)

dense_rankings = []
bm25_rankings = []
hybrid_rankings = []

start = time.perf_counter()

for position, item in enumerate(
    evaluation,
    start=1,
):
    query = item["query"]

    query_vector = model.encode(
        [
            "query: " + query
        ],
        normalize_embeddings=True,
    )[0]

    dense_scores = (
        query_vector
        @ doc_vectors.T
    )

    dense_order = np.argsort(
        -dense_scores
    )

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_order = np.argsort(
        -bm25_scores
    )

    hybrid_order = reciprocal_rank_fusion(
        dense_order,
        bm25_order,
    )

    dense_rankings.append(
        dense_order
    )

    bm25_rankings.append(
        bm25_order
    )

    hybrid_rankings.append(
        hybrid_order
    )

    if position % 50 == 0:
        print(
            f"Processed {position}/"
            f"{len(evaluation)} queries"
        )

elapsed = time.perf_counter() - start

print()
print(
    f"Retrieval evaluation completed "
    f"in {elapsed:.2f}s"
)
print()

dense_summary, dense_results = evaluate(
    "dense",
    dense_rankings,
    evaluation,
)

bm25_summary, bm25_results = evaluate(
    "bm25",
    bm25_rankings,
    evaluation,
)

hybrid_summary, hybrid_results = evaluate(
    "hybrid",
    hybrid_rankings,
    evaluation,
)

summaries = [
    dense_summary,
    bm25_summary,
    hybrid_summary,
]

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

for summary in summaries:
    print(
        f"{summary['retriever']:<12}"
        f"{summary['recall@1']:>10.4f}"
        f"{summary['recall@3']:>10.4f}"
        f"{summary['recall@5']:>10.4f}"
        f"{summary['mrr']:>10.4f}"
        f"{summary['ndcg@5']:>12.4f}"
    )

print("=" * 90)

output = {
    "evaluation_dataset": str(EVAL_PATH),
    "kb_path": str(KB_PATH),
    "model": MODEL,
    "query_count": len(evaluation),
    "kb_count": len(articles),
    "summaries": summaries,
    "per_query": {
        "dense": dense_results,
        "bm25": bm25_results,
        "hybrid": hybrid_results,
    },
}

output_path = Path(
    "data/eval/retrieval_results.json"
)

output_path.write_text(
    json.dumps(
        output,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print(
    f"Detailed results written to: "
    f"{output_path}"
)