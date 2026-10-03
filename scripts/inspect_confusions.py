import json
from pathlib import Path

from collections import Counter

articles = [
    json.loads(line)
    for line in Path("data/synthetic/kb_articles.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]

tickets = [
    json.loads(line)
    for line in Path("data/synthetic/tickets.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]

article_by_id = {a["id"]: a for a in articles}

confusions = Counter()

for ticket in tickets:
    true_class = ticket["class_id"]
    kb_id = ticket["kb_id"]

    article = article_by_id.get(kb_id)

    if not article:
        continue

    # We only inspect the major confusing classes.
    if true_class in {
        "speed.degraded",
        "connectivity.outage",
        "device.hardware",
        "account.provisioning",
    }:
        print("\n" + "=" * 70)
        print("TRUE CLASS:", true_class)
        print("KB ID:", kb_id)
        print("COMPLAINT:", ticket["complaint"])
        print("ARTICLE TITLE:", article["title"])
        print("ARTICLE CLASS:", article["class_id"])
        print("SYMPTOMS:")

        for symptom in article["symptoms"]:
            print(" -", symptom)

        print("ROOT CAUSE:")
        print(article["root_cause"])