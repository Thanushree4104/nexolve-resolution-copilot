import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.pii import redact

MODEL = "intfloat/multilingual-e5-small"


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


articles = read_jsonl("data/synthetic/kb_articles.jsonl")
tickets = read_jsonl("data/synthetic/tickets.jsonl")
ids = [a["id"] for a in articles]
classes = [a["class_id"] for a in articles]

VARIANTS = {
    "one vector per article": lambda a: [a["title"] + ". " + " ".join(a["symptoms"])],
    "per symptom, with title": lambda a: [f"{a['title']}. {s}" for s in a["symptoms"]],
    "per symptom, no title": lambda a: list(a["symptoms"]),
}

model = SentenceTransformer(MODEL)
q = model.encode(
    ["query: " + redact(t["complaint"]).text for t in tickets],
    normalize_embeddings=True,
    batch_size=32,
)
subsets = {
    "all": list(range(len(tickets))),
    "en": [i for i, t in enumerate(tickets) if t["language"] == "en"],
    "de": [i for i, t in enumerate(tickets) if t["language"] == "de"],
}

for name, fn in VARIANTS.items():
    texts, owner = [], []
    for j, a in enumerate(articles):
        for chunk in fn(a):
            texts.append("passage: " + chunk)
            owner.append(j)
    owner = np.array(owner)
    sims = q @ model.encode(texts, normalize_embeddings=True).T
    scores = np.stack([sims[:, owner == j].max(axis=1) for j in range(len(articles))], axis=1)

    print(f"\n{name}  ({len(texts)} vectors)")
    for label, subset in subsets.items():
        ranks, cls = [], []
        for i in subset:
            order = np.argsort(-scores[i])
            ranks.append([ids[j] for j in order].index(tickets[i]["kb_id"]))
            cls.append(classes[order[0]] == tickets[i]["class_id"])
        r = np.array(ranks)
        print(
            f"  {label:<4} n={len(subset):<4} R@1={(r<1).mean():.2f}  R@3={(r<3).mean():.2f}  "
            f"R@5={(r<5).mean():.2f}  MRR={(1/(r+1)).mean():.2f}  class@1={np.mean(cls):.2f}"
        )