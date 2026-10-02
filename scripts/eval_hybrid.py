import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.pii import redact

MODEL = "intfloat/multilingual-e5-small"
RRF_K = 60


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


articles = read_jsonl("data/synthetic/kb_articles.jsonl")
tickets = read_jsonl("data/synthetic/tickets.jsonl")
ids = [a["id"] for a in articles]
classes = [a["class_id"] for a in articles]


def doc_text(a):
    return (
        a["title"] + ". " + " ".join(a["symptoms"]) + " "
        + " ".join(d["q"] for d in a["diagnostic_questions"]) + " "
        + " ".join(s["text"] for s in a["steps"])
    )


def tok(text):
    return re.findall(r"\w+", text.lower())


model = SentenceTransformer(MODEL)
doc_vecs = model.encode(["passage: " + doc_text(a) for a in articles], normalize_embeddings=True)
texts = [redact(t["complaint"]).text for t in tickets]
q_vecs = model.encode(["query: " + x for x in texts], normalize_embeddings=True, batch_size=32)
bm25 = BM25Okapi([tok(doc_text(a)) for a in articles])


def rrf(*orders):
    fused = np.zeros(len(articles))
    for order in orders:
        for rank, j in enumerate(order):
            fused[j] += 1.0 / (RRF_K + rank + 1)
    return np.argsort(-fused)


orders = {"dense": [], "bm25": [], "hybrid": []}
for i, text in enumerate(texts):
    d = np.argsort(-(q_vecs[i] @ doc_vecs.T))
    b = np.argsort(-bm25.get_scores(tok(text)))
    orders["dense"].append(d)
    orders["bm25"].append(b)
    orders["hybrid"].append(rrf(d, b))

subsets = {
    "all": list(range(len(tickets))),
    "en": [i for i, t in enumerate(tickets) if t["language"] == "en"],
    "de": [i for i, t in enumerate(tickets) if t["language"] == "de"],
}


def metrics(method, subset):
    ranks, cls = [], []
    for i in subset:
        order = orders[method][i]
        ranks.append([ids[j] for j in order].index(tickets[i]["kb_id"]))
        cls.append(classes[order[0]] == tickets[i]["class_id"])
    r = np.array(ranks)
    return (r < 1).mean(), (r < 3).mean(), (r < 5).mean(), (1 / (r + 1)).mean(), np.mean(cls)


print(f"{'method':<8}{'subset':<6}{'n':<5}{'R@1':<6}{'R@3':<6}{'R@5':<6}{'MRR':<6}{'class@1'}")
for method in orders:
    for label, subset in subsets.items():
        m = metrics(method, subset)
        print(f"{method:<8}{label:<6}{len(subset):<5}" + "".join(f"{v:<6.2f}" for v in m))

conf = Counter()
for i, order in enumerate(orders["dense"]):
    pred = classes[order[0]]
    if pred != tickets[i]["class_id"]:
        conf[(tickets[i]["class_id"], pred)] += 1
print("\nMost common dense confusions (true -> predicted):")
for (t, p), n in conf.most_common(6):
    print(f"  {n:>3}  {t} -> {p}")