import json
from pathlib import Path

KEYWORDS = ["restart", "reboot", "power cycl", "reset", "unplug", "cable", "switched off", "turned off"]
rows = [
    json.loads(l)
    for l in Path("data/synthetic/tickets.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
]
en = [r for r in rows if r["language"] == "en"]
mentions = [r for r in en if any(k in r["complaint"].lower() for k in KEYWORDS)]
missing = [r for r in mentions if not r["steps_already_tried"]]
print(f"English tickets: {len(en)}")
print(f"Complaint mentions a tried step: {len(mentions)}")
print(f"...but steps_already_tried is empty: {len(missing)}")