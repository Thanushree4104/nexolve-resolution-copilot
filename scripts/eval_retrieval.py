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
article_ids = [a["id"] for a in articles]
article_classes = [a["class_id"] for a in articles]

# Two ways of turning a KB article into text to embed.
VARIANTS = {
    "title+symptoms": lambda a: a["title"] + ". " + " ".join(a["symptoms"]),
    "title+symptoms+steps": lambda a: (
        a["title"] + ". " + " ".join(a["symptoms"]) + " "
        + " ".join(d["q"] for d in a["diagnostic_questions"]) + " "
        + " ".join(s["text"] for s in a["steps"])
    ),
}

model = SentenceTransformer(MODEL)
queries = model.encode(
    ["query: " + redact(t["complaint"]).text for t in tickets],
    normalize_embeddings=True,
    batch_size=32,
)


def evaluate(doc_vecs, subset):
    ranks, class_hits = [], []
    for idx in subset:
        t = tickets[idx]
        order = np.argsort(-(queries[idx] @ doc_vecs.T))
        ranks.append([article_ids[j] for j in order].index(t["kb_id"]))
        class_hits.append(article_classes[order[0]] == t["class_id"])
    r = np.array(ranks)
    return {
        "n": len(r),
        "R@1": (r < 1).mean(),
        "R@3": (r < 3).mean(),
        "R@5": (r < 5).mean(),
        "MRR": (1 / (r + 1)).mean(),
        "class@1": float(np.mean(class_hits)),
    }


subsets = {
    "all": list(range(len(tickets))),
    "en": [i for i, t in enumerate(tickets) if t["language"] == "en"],
    "de": [i for i, t in enumerate(tickets) if t["language"] == "de"],
}

for name, fn in VARIANTS.items():
    vecs = model.encode(["passage: " + fn(a) for a in articles], normalize_embeddings=True)
    print(f"\n{name}")
    for label, subset in subsets.items():
        m = evaluate(vecs, subset)
        cols = "  ".join(f"{k}={m[k]:.2f}" for k in ("R@1", "R@3", "R@5", "MRR", "class@1"))
        print(f"  {label:<4} n={m['n']:<4} {cols}")