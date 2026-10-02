import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"

t0 = time.perf_counter()
model = SentenceTransformer(MODEL)
print(f"model loaded in {time.perf_counter() - t0:.1f}s")

articles = [
    json.loads(line)
    for line in Path("data/synthetic/kb_articles.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def doc_text(a):
    return f"{a['title']}. " + " ".join(a["symptoms"])


# e5 models expect these prefixes: "passage: " for documents, "query: " for searches.
t0 = time.perf_counter()
docs = model.encode(
    ["passage: " + doc_text(a) for a in articles], normalize_embeddings=True, batch_size=16
)
print(f"embedded {len(articles)} articles in {time.perf_counter() - t0:.1f}s")

queries = [
    "My broadband drops every evening around 8 and I've already restarted the router twice",
    "Mein Internet bricht jeden Abend gegen 20 Uhr ab",
    "I was charged twice on my bill this month",
]
t0 = time.perf_counter()
qs = model.encode(["query: " + q for q in queries], normalize_embeddings=True)
print(f"embedded {len(queries)} queries in {(time.perf_counter() - t0) * 1000:.0f}ms")

scores = qs @ docs.T
for q, row in zip(queries, scores):
    print("\nQUERY:", q)
    for i in row.argsort()[::-1][:3]:
        a = articles[i]
        print(f"  {row[i]:.3f}  {a['id']}  {a['class_id']}  | {a['root_cause']}")