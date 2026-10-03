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

CHECKPOINT = Path(
    f".cache/evaluation/parser_eval_seed_{SEED}.jsonl"
)


def load_checkpoint():
    results = {}

    if not CHECKPOINT.exists():
        return results

    for line in CHECKPOINT.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        if row.get("status") == "success":
            results[row["ticket_id"]] = row

    return results


def save_result(ticket, parsed):
    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "ticket_id": ticket["id"],
        "true_class": ticket["class_id"],
        "predicted_class": parsed.category,
        "confidence": parsed.category_confidence,
        "status": "success",
    }

    with CHECKPOINT.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )

    return row


def main():

    if settings.llm_provider != "groq":
        sys.exit(
            'Set the provider first: '
            '$env:LLM_PROVIDER="groq"'
        )

    tickets = [
        json.loads(line)
        for line in Path(
            "data/synthetic/tickets.jsonl"
        )
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]

    sample = random.Random(SEED).sample(
        tickets,
        N,
    )

    checkpoint = load_checkpoint()

    print(
        f"Loaded {len(checkpoint)} "
        f"cached evaluations from checkpoint."
    )

    parser = ComplaintParser(
        get_provider(),
        load_taxonomy(),
    )

    rows = []

    for i, ticket in enumerate(sample, 1):

        ticket_id = ticket["id"]

        # --------------------------------------------------
        # EXISTING CHECKPOINT
        # --------------------------------------------------

        if ticket_id in checkpoint:

            row = checkpoint[ticket_id]

            rows.append(
                (
                    row["true_class"],
                    row["predicted_class"],
                    row["confidence"],
                )
            )

            print(
                f"  {i}/{N}  "
                f"{ticket_id} "
                f"[cached evaluation]"
            )

            continue

        # --------------------------------------------------
        # NEW LLM EVALUATION
        # --------------------------------------------------

        try:

            parsed = parser.parse(
                ticket["complaint"]
            )

            row = save_result(
                ticket,
                parsed,
            )

            rows.append(
                (
                    row["true_class"],
                    row["predicted_class"],
                    row["confidence"],
                )
            )

            print(
                f"  {i}/{N}  "
                f"{ticket_id} "
                f"[evaluated]"
            )

        except (
            LLMError,
            LLMUnavailableError,
        ) as e:

            print(
                f"  {i}/{N}  "
                f"{ticket_id} "
                f"[failed: {str(e)[:120]}]"
            )

            print()
            print(
                "Evaluation paused safely."
            )
            print(
                "Completed results have been saved."
            )
            print(
                "Run the same command again later "
                "to resume."
            )

            break

        # Gentle pacing for NEW requests only.
        time.sleep(2)

    # ------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------

    if not rows:
        print(
            "\nNo evaluation results available."
        )
        return

    correct = [
        r
        for r in rows
        if r[0] == r[1]
    ]

    wrong = [
        r
        for r in rows
        if r[0] != r[1]
    ]

    print()
    print(
        f"parsed {len(rows)} of {N} "
        f"(checkpoint + new evaluations), "
        f"seed {SEED}"
    )

    print(
        f"category accuracy: "
        f"{len(correct) / len(rows):.2f}"
    )

    print(
        f"unknown rate:      "
        f"{sum(r[1] == 'unknown' for r in rows) / len(rows):.2f}"
    )

    if correct and wrong:

        right_confidence = (
            sum(r[2] for r in correct)
            / len(correct)
        )

        wrong_confidence = (
            sum(r[2] for r in wrong)
            / len(wrong)
        )

        print(
            f"mean confidence:   "
            f"right={right_confidence:.2f} "
            f"wrong={wrong_confidence:.2f}"
        )

    print()
    print(
        "accuracy by true class:"
    )

    for cls in sorted(
        {r[0] for r in rows}
    ):

        subset = [
            r
            for r in rows
            if r[0] == cls
        ]

        correct_count = sum(
            r[0] == r[1]
            for r in subset
        )

        print(
            f"  {cls:<26} "
            f"{correct_count}/{len(subset)}"
        )

    print()
    print(
        "top confusions "
        "(true -> predicted):"
    )

    for (
        true_class,
        predicted_class,
    ), count in Counter(
        (r[0], r[1])
        for r in wrong
    ).most_common(5):

        print(
            f"  {count:>3}  "
            f"{true_class} -> "
            f"{predicted_class}"
        )


if __name__ == "__main__":
    main()