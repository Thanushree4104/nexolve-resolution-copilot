import json
import re
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
classes = np.array([a["class_id"] for a in articles])


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


def rank_of_true(i, method):
    cand = np.where(classes == tickets[i]["class_id"])[0]
    d_order = np.argsort(-(q_vecs[i] @ doc_vecs[cand].T))
    b_order = np.argsort(-bm25.get_scores(tok(texts[i]))[cand])
    if method == "dense":
        order = d_order
    elif method == "bm25":
        order = b_order
    else:
        fused = np.zeros(len(cand))
        for o in (d_order, b_order):
            for rank, j in enumerate(o):
                fused[j] += 1.0 / (RRF_K + rank + 1)
        order = np.argsort(-fused)
    return [ids[cand[j]] for j in order].index(tickets[i]["kb_id"])


subsets = {
    "all": list(range(len(tickets))),
    "en": [i for i, t in enumerate(tickets) if t["language"] == "en"],
    "de": [i for i, t in enumerate(tickets) if t["language"] == "de"],
}

print("Candidates restricted to the TRUE class (4 articles each).")
print("Random guessing would score: R@1=0.25  R@2=0.50  MRR=0.52\n")
print(f"{'method':<8}{'subset':<6}{'n':<5}{'R@1':<6}{'R@2':<6}{'MRR'}")
for method in ("dense", "bm25", "hybrid"):
    for label, subset in subsets.items():
        r = np.array([rank_of_true(i, method) for i in subset])
        print(f"{method:<8}{label:<6}{len(subset):<5}{(r<1).mean():<6.2f}{(r<2).mean():<6.2f}{(1/(r+1)).mean():.2f}")