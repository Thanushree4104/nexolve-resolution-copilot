import json
import random
from pathlib import Path

from app.core.taxonomy import load_taxonomy
from app.llm.factory import get_provider
from app.services.parser import ComplaintParser

tickets = [
    json.loads(l)
    for l in Path("data/synthetic/tickets.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
]
sample = random.Random(11).sample(tickets, 100)  # same sample as eval_parser, so calls are cached
parser = ComplaintParser(get_provider(), load_taxonomy())

for t in sample:
    p = parser.parse(t["complaint"])
    if p.category != t["class_id"]:
        print("-" * 70)
        print("COMPLAINT:", t["complaint"])
        print(f"TRUE: {t['class_id']}  (cause: {t['root_cause']})")
        print(f"GOT:  {p.category}  (symptoms: {p.symptoms})")