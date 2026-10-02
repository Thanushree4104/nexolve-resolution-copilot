import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from app.core.config import settings
from app.core.taxonomy import load_taxonomy
from app.llm.base import LLMError, LLMUnavailableError
from app.llm.factory import get_provider
from app.services.parser import ComplaintParser

N = 100
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 11


def parse_with_retry(parser, text):
    try:
        return parser.parse(text)
    except LLMUnavailableError:
        time.sleep(30)
        return parser.parse(text)


def main():
    if settings.llm_provider != "groq":
        sys.exit('Set the provider first:  $env:LLM_PROVIDER="groq"')
    tickets = [
        json.loads(l)
        for l in Path("data/synthetic/tickets.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    sample = random.Random(SEED).sample(tickets, N)
    parser = ComplaintParser(get_provider(), load_taxonomy())

    rows, failed = [], 0
    for i, t in enumerate(sample, 1):
        try:
            p = parse_with_retry(parser, t["complaint"])
        except (LLMError, LLMUnavailableError) as e:
            failed += 1
            print(f"  ticket {t['id']} failed: {str(e)[:100]}")
            continue
        rows.append((t["class_id"], p.category, p.category_confidence))
        if i % 10 == 0:
            print(f"  {i}/{N}")
        time.sleep(1)  # gentle on free-tier limits

    correct = [r for r in rows if r[0] == r[1]]
    print(f"\nparsed {len(rows)} of {N} (failed: {failed}), seed {SEED}")
    print(f"category accuracy: {len(correct) / len(rows):.2f}")
    print(f"unknown rate:      {sum(r[1] == 'unknown' for r in rows) / len(rows):.2f}")
    wrong = [r for r in rows if r[0] != r[1]]
    if correct and wrong:
        print(
            f"mean confidence:   right={sum(r[2] for r in correct) / len(correct):.2f}"
            f"  wrong={sum(r[2] for r in wrong) / len(wrong):.2f}"
        )
    print("\naccuracy by true class:")
    for cls in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == cls]
        print(f"  {cls:<26} {sum(r[0] == r[1] for r in sub)}/{len(sub)}")
    print("\ntop confusions (true -> predicted):")
    for (a, b), n in Counter((r[0], r[1]) for r in wrong).most_common(5):
        print(f"  {n:>3}  {a} -> {b}")


if __name__ == "__main__":
    main()