import json
from collections import Counter
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("data/synthetic/tickets.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print("tickets:", len(rows))
for key in ("class_id", "language", "resolution_quality", "reopened"):
    print(key, dict(Counter(r[key] for r in rows)))
print("avg complaint chars:", sum(len(r["complaint"]) for r in rows) // len(rows))