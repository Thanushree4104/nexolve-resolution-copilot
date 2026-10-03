import json
from pathlib import Path


TICKETS_PATH = Path("data/synthetic/tickets.jsonl")
KB_PATH = Path("data/synthetic/kb_articles.jsonl")
HELDOUT_PATH = Path("data/synthetic/kb_heldback.jsonl")
OUTPUT_PATH = Path("data/eval/retrieval_eval.jsonl")


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main():
    tickets = load_jsonl(TICKETS_PATH)
    kb_articles = load_jsonl(KB_PATH)
    heldout_articles = load_jsonl(HELDOUT_PATH)

    active_kb_ids = {
        article["id"]
        for article in kb_articles
    }

    heldout_ids = {
        article["id"]
        for article in heldout_articles
    }

    evaluation_records = []

    skipped_heldout = 0
    skipped_unknown = 0

    for ticket in tickets:
        kb_id = ticket.get("kb_id")
        complaint = ticket.get("complaint", "").strip()

        if not kb_id or not complaint:
            continue

        if kb_id in heldout_ids:
            skipped_heldout += 1
            continue

        if kb_id not in active_kb_ids:
            skipped_unknown += 1
            continue

        evaluation_records.append(
            {
                "ticket_id": ticket["id"],
                "query": complaint,
                "relevant_ids": [kb_id],
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in evaluation_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("Retrieval evaluation dataset created.")
    print(f"Total tickets: {len(tickets)}")
    print(f"Evaluation queries: {len(evaluation_records)}")
    print(f"Skipped held-back KB tickets: {skipped_heldout}")
    print(f"Skipped unknown KB tickets: {skipped_unknown}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()