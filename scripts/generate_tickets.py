import json
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from pydantic import ValidationError

from app.core.config import settings
from app.core.taxonomy import load_taxonomy
from app.llm.base import LLMError, LLMUnavailableError
from app.llm.factory import get_provider
from app.schemas.ticket import Ticket, parse_batch

KB_PATH = Path("data/synthetic/kb_articles.jsonl")
OUT = Path("data/synthetic/tickets.jsonl")
TODAY = date(2026, 10, 1)
N_TICKETS = 400
BATCH = 4

PERSONAS = ["remote worker", "elderly user", "small shop owner", "student", "gamer", "busy parent"]
STYLES = ["terse", "rambling", "angry", "technical", "polite", "non-native speaker"]
LANGS = {"en": 0.85, "de": 0.15}  # Hindi and Tamil get added here later
TRIED_POOL = [
    "restarted the router twice",
    "checked all the cables",
    "tried a different device",
    "reset the router to factory settings",
    "waited 24 hours",
    "switched the Wi-Fi off and on",
]
REGIONS = ["Chennai", "Bengaluru", "Hyderabad", "Mumbai", "Delhi", "Pune", "Kolkata", "Coimbatore"]
FIRMWARE = ["2.1.4", "2.2.0", "2.3.1", "3.0.2"]
QUALITY = {"detailed": 0.55, "brief": 0.30, "vague": 0.15}
REOPEN_RATE = {"detailed": 0.03, "brief": 0.10, "vague": 0.30}  # lazy notes reopen more

PROMPT = """You generate realistic resolved support tickets for a fictional broadband and
mobile operator called Nimbus. Use fictional product names only.

Write {n} different tickets, one for each seed below.

{seeds}

Return JSON only, in this shape, with exactly {n} items in the same order as the seeds:
{{"tickets": [{{"complaint": string, "resolution_notes": string, "steps_taken": [string]}}]}}

Rules:
- complaint: the customer's raw message, 2-6 sentences, in the seed's language and wording
  style. Customers describe symptoms, not diagnoses, and must NOT reuse the words of the
  root cause.
- If steps_already_tried is not empty, the customer mentions having done those things.
- resolution_notes follows resolution_quality: detailed = 3-5 sentences of specific actions
  and the outcome; brief = 1-2 sentences; vague = a lazy note under 8 words such as "fixed"
  or "sorted after reset".
- steps_taken: the concrete actions taken, or an empty list when the note is vague.
- Never include real names, phone numbers, emails or addresses."""


def load_kb_index():
    if not KB_PATH.exists():
        sys.exit("Run scripts.generate_kb first.")
    index = {}
    for line in KB_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            a = json.loads(line)
            index[(a["class_id"], a["root_cause"])] = (a["id"], a["product"])
    return index


def build_seeds(tax, kb_index):
    rng = random.Random(7)  # fixed seed: the same tasks every run
    classes = tax.active_classes()
    seeds = []
    for i in range(N_TICKETS):
        c = classes[i % len(classes)]
        cause = rng.choice(c.root_causes)
        kb_id, product = kb_index[(c.id, cause)]
        quality = rng.choices(list(QUALITY), weights=list(QUALITY.values()))[0]
        reopened = rng.random() < REOPEN_RATE[quality]
        tried = rng.sample(TRIED_POOL, rng.choice([0, 0, 1, 2]))
        seeds.append(
            {
                "id": f"T-{10001 + i}",
                "class_id": c.id,
                "class_name": c.name,
                "product": product,
                "root_cause": cause,
                "kb_id": kb_id,
                "language": rng.choices(list(LANGS), weights=list(LANGS.values()))[0],
                "style": rng.choice(STYLES),
                "persona": rng.choice(PERSONAS),
                "resolution_quality": quality,
                "steps_already_tried": tried,
                "reopened": reopened,
                "first_contact_resolution": (not reopened) and rng.random() < 0.8,
                "created_at": (TODAY - timedelta(days=rng.randint(0, 365))).isoformat(),
                "region": rng.choice(REGIONS),
                "firmware": rng.choice(FIRMWARE) if product in ("Nimbus ONT-5G", "Nimbus Mesh") else None,
            }
        )
    return seeds


def seed_line(n, s):
    return (
        f"{n}. class: {s['class_name']}; root_cause: {s['root_cause']}; product: {s['product']}; "
        f"customer: {s['persona']}; wording_style: {s['style']}; language: {s['language']}; "
        f"steps_already_tried: {s['steps_already_tried'] or 'none'}; "
        f"resolution_quality: {s['resolution_quality']}"
    )


def generate_batch(provider, batch):
    prompt = PROMPT.format(
        n=len(batch), seeds="\n".join(seed_line(i + 1, s) for i, s in enumerate(batch))
    )
    for attempt in range(3):
        try:
            resp = provider.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.8 + 0.1 * attempt,  # a retry must not replay a cached bad answer
                max_tokens=6000,
                json_mode=True,
            )
            return parse_batch(resp.text, len(batch)), resp.cached
        except LLMUnavailableError as e:
            wait = 20 * (attempt + 1)
            print(f"  LLM unavailable ({str(e)[:150]}), waiting {wait}s...")
            time.sleep(wait)
        except (ValueError, ValidationError, LLMError) as e:
            print(f"  attempt {attempt + 1} failed: {str(e)[:120]}")
    return None, False


def main():
    if settings.llm_provider != "groq":
        sys.exit('Set the provider first:  $env:LLM_PROVIDER="groq"')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    provider = get_provider()
    seeds = build_seeds(load_taxonomy(), load_kb_index())
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    failed = []

    for i in range(0, len(seeds), BATCH):
        batch = seeds[i : i + BATCH]
        if all(s["id"] in done for s in batch):
            continue
        print(f"{batch[0]['id']} .. {batch[-1]['id']}")
        texts, cached = generate_batch(provider, batch)
        if texts is None:
            failed.append(batch[0]["id"])
            continue
        with OUT.open("a", encoding="utf-8") as f:
            for seed, text in zip(batch, texts):
                seed = {k: v for k, v in seed.items() if k != "class_name"}
                f.write(Ticket(**seed, **text.model_dump()).model_dump_json() + "\n")
        if not cached:
            time.sleep(2)

    print(f"\nDone. Failed batches (rerun to retry): {failed or 'none'}")


if __name__ == "__main__":
    main()