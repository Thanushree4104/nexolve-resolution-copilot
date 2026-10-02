import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.pii import redact

MODEL = "intfloat/multilingual-e5-small"
PAIRS = [
    ("speed.degraded", "connectivity.intermittent"),
    ("connectivity.outage", "connectivity.intermittent"),
]


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


articles = read_jsonl("data/synthetic/kb_articles.jsonl")
tickets = read_jsonl("data/synthetic/tickets.jsonl")
by_id = {a["id"]: a for a in articles}


def doc_text(a):
    return (
        a["title"] + ". " + " ".join(a["symptoms"]) + " "
        + " ".join(d["q"] for d in a["diagnostic_questions"]) + " "
        + " ".join(s["text"] for s in a["steps"])
    )


model = SentenceTransformer(MODEL)
docs = model.encode(["passage: " + doc_text(a) for a in articles], normalize_embeddings=True)
rng = random.Random(3)

for true_c, pred_c in PAIRS:
    wrong = []
    for t in tickets:
        if t["class_id"] != true_c:
            continue
        q = model.encode("query: " + redact(t["complaint"]).text, normalize_embeddings=True)
        top = int(np.argmax(q @ docs.T))
        if articles[top]["class_id"] == pred_c:
            wrong.append((t, articles[top]))
    print("=" * 70)
    print(f"{true_c} mistaken for {pred_c}: {len(wrong)} tickets")
    for t, got in rng.sample(wrong, min(3, len(wrong))):
        want = by_id[t["kb_id"]]
        print("\nCOMPLAINT:", t["complaint"])
        print(f"SHOULD BE: {want['id']} {want['title']}  (cause: {want['root_cause']})")
        print(f"GOT:       {got['id']} {got['title']}  (cause: {got['root_cause']})")