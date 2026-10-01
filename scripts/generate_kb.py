import json
import random
import sys
import time
import zlib
from datetime import date, timedelta
from pathlib import Path

from pydantic import ValidationError

from app.core.config import settings
from app.core.taxonomy import load_taxonomy
from app.llm.base import LLMError, LLMUnavailableError
from app.llm.factory import get_provider
from app.schemas.kb import KBArticle, parse_body

OUT_ACTIVE = Path("data/synthetic/kb_articles.jsonl")
OUT_HELD = Path("data/synthetic/kb_heldback.jsonl")
TODAY = date(2026, 10, 1)

PROMPT = """You write internal knowledge base articles for the support desk of a fictional
broadband and mobile operator called Nimbus. Use only fictional product names. Do not
include real phone numbers, URLs or brand names.

Write one article for this case:
- Problem class: {class_name} ({description})
- Root cause this article covers: {root_cause}
- Product: {product}

Return JSON only, with exactly these keys:
{{
  "title": string,
  "symptoms": [string],              // 3-5 items, how customers describe it in everyday words
  "diagnostic_questions": [{{"q": string, "if_yes": string, "if_no": string}}],   // 2-3 items
  "steps": [{{"n": int, "text": string}}],    // 5-8 concrete, ordered steps with the expected outcome of each
  "escalate_when": [string],         // 1-3 conditions
  "notes": string
}}

Rules:
- The symptoms must NOT reuse the exact words of the root cause or the title, so keyword
  search alone cannot match them.
- Steps must be specific and technical, written for a support agent.
- Do not tell the agent to restart the router as the only fix."""


def load_done(*paths):
    done = set()

    for p in paths:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["id"])

    return done


def build_tasks(tax):
    tasks, n = [], 1000

    for c in tax.classes:
        for i, cause in enumerate(c.root_causes):
            n += 1
            tasks.append(
                (
                    f"KB-{n}",
                    c,
                    cause,
                    c.products[i % len(c.products)],
                )
            )

    return tasks


def generate_one(provider, task, rng):
    kb_id, c, cause, product = task

    prompt = PROMPT.format(
        class_name=c.name,
        description=c.description,
        root_cause=cause,
        product=product,
    )

    for attempt in range(3):
        try:
            # Temperature changes per attempt so a retry doesn't replay
            # a cached bad answer.
            resp = provider.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.6 + 0.1 * attempt,
                max_tokens=4000,
                json_mode=True,
            )

            body = parse_body(resp.text)

            reviewed = TODAY - timedelta(
                days=rng.randint(10, 540)
            )

            article = KBArticle(
                **body.model_dump(),
                id=kb_id,
                class_id=c.id,
                product=product,
                root_cause=cause,
                last_reviewed=reviewed.isoformat(),
                status="active" if c.status == "active" else "held_back",
            )

            return article, resp.cached

        except LLMUnavailableError:
            wait = 20 * (attempt + 1)
            print(f"  LLM unavailable, waiting {wait}s...")
            time.sleep(wait)

        except (ValueError, ValidationError, LLMError) as e:
            print(
                f"  attempt {attempt + 1} failed: {str(e)[:120]}"
            )

    return None, False


def main():
    if settings.llm_provider != "groq":
        sys.exit(
            'Set the provider first:  $env:LLM_PROVIDER="groq"'
        )

    OUT_ACTIVE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    provider = get_provider()
    tax = load_taxonomy()

    done = load_done(
        OUT_ACTIVE,
        OUT_HELD,
    )

    tasks = build_tasks(tax)
    failed = []

    for task in tasks:
        kb_id, c, cause, _ = task

        # Keep dates stable regardless of what was already done.
        # zlib.crc32() is deterministic across Python runs,
        # unlike Python's built-in hash() for strings.
        task_rng = random.Random(
            zlib.crc32(kb_id.encode("utf-8"))
        )

        if kb_id in done:
            continue

        print(
            f"{kb_id}  {c.id}  ({cause})"
        )

        article, cached = generate_one(
            provider,
            task,
            task_rng,
        )

        if article is None:
            failed.append(kb_id)
            continue

        out = (
            OUT_ACTIVE
            if c.status == "active"
            else OUT_HELD
        )

        with out.open("a", encoding="utf-8") as f:
            f.write(
                article.model_dump_json() + "\n"
            )

        if not cached:
            # Be gentle with free-tier rate limits.
            time.sleep(2)

    done = load_done(
        OUT_ACTIVE,
        OUT_HELD,
    )

    print(
        f"\nDone: {len(done)} of {len(tasks)} articles saved."
    )

    if failed:
        print(
            "Failed (rerun the script to retry):",
            failed,
        )


if __name__ == "__main__":
    main()