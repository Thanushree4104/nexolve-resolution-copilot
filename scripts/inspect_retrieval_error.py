import json
from pathlib import Path

from app.services.retriever import HybridRetriever


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


tickets = read_jsonl("data/synthetic/tickets.jsonl")
articles = read_jsonl("data/synthetic/kb_articles.jsonl")

article_by_id = {
    article["id"]: article
    for article in articles
}

retriever = HybridRetriever()

errors = []

for ticket in tickets:
    results = retriever.search(
        ticket["complaint"],
        top_k=3,
    )

    predicted = results[0]

    if predicted["id"] != ticket["kb_id"]:
        correct = article_by_id[ticket["kb_id"]]

        errors.append(
            {
                "complaint": ticket["complaint"],
                "true_kb": ticket["kb_id"],
                "true_class": ticket["class_id"],
                "predicted_kb": predicted["id"],
                "predicted_class": predicted["class_id"],
                "predicted_title": predicted["title"],
                "correct_title": correct["title"],
                "top3": [
                    (
                        r["id"],
                        r["class_id"],
                        round(r["dense_score"], 3),
                        round(r["bm25_score"], 3),
                    )
                    for r in results
                ],
            }
        )


print(f"\nTotal retrieval errors: {len(errors)}")
print("=" * 100)

for i, error in enumerate(errors[:20], start=1):
    print(f"\nERROR {i}")
    print("-" * 100)

    print("Complaint:")
    print(error["complaint"])

    print("\nTRUE:")
    print(error["true_kb"], "|", error["true_class"])
    print(error["correct_title"])

    print("\nPREDICTED:")
    print(error["predicted_kb"], "|", error["predicted_class"])
    print(error["predicted_title"])

    print("\nTOP 3:")
    for item in error["top3"]:
        print(item)