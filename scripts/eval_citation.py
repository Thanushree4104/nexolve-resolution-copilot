import json
import re
from pathlib import Path


INPUT_FILE = Path("data/eval/answer_results.jsonl")
OUTPUT_FILE = Path("data/eval/citation_results.json")


CITATION_PATTERN = re.compile(
    r"\[KB-[A-Za-z0-9_-]+\]"
)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def extract_citations(answer):
    return CITATION_PATTERN.findall(answer or "")


def unique(values):
    return list(dict.fromkeys(values))


def evaluate_record(record):
    answer = record.get("answer") or ""

    retrieved_ids = set(
        record.get("retrieved_ids") or []
    )

    citations = extract_citations(answer)

    citation_ids = [
        citation[1:-1]
        for citation in citations
    ]

    unique_citation_ids = unique(
        citation_ids
    )

    unsupported_citations = [
        citation
        for citation in unique_citation_ids
        if citation not in retrieved_ids
    ]

    valid_citations = [
        citation
        for citation in unique_citation_ids
        if citation in retrieved_ids
    ]

    has_answer = bool(answer.strip())
    has_citation = bool(citations)

    citation_valid = (
        has_citation
        and not unsupported_citations
    )

    error = record.get("error")

    if error:
        status = "error"

    elif not has_answer:
        status = "empty_answer"

    elif not has_citation:
        status = "no_citations"

    elif unsupported_citations:
        status = "unsupported_citations"

    else:
        status = "passed"

    return {
        "ticket_id": record.get("ticket_id"),
        "query": record.get("query"),
        "retrieved_ids": sorted(
            retrieved_ids
        ),
        "citations": unique_citation_ids,
        "valid_citations": valid_citations,
        "unsupported_citations": (
            unsupported_citations
        ),
        "has_answer": has_answer,
        "has_citation": has_citation,
        "citation_valid": citation_valid,
        "status": status,
        "error": error,
    }


def calculate_summary(results):
    total = len(results)

    passed = sum(
        r["status"] == "passed"
        for r in results
    )

    errors = sum(
        r["status"] == "error"
        for r in results
    )

    empty_answers = sum(
        r["status"] == "empty_answer"
        for r in results
    )

    no_citations = sum(
        r["status"] == "no_citations"
        for r in results
    )

    unsupported = sum(
        r["status"] == "unsupported_citations"
        for r in results
    )

    answers_available = sum(
        r["has_answer"]
        for r in results
    )

    citation_present = sum(
        r["has_citation"]
        for r in results
    )

    valid_citation_records = sum(
        r["citation_valid"]
        for r in results
    )

    unsupported_citation_count = sum(
        len(r["unsupported_citations"])
        for r in results
    )

    return {
        "total_records": total,
        "answers_available": answers_available,
        "records_with_citations": citation_present,
        "records_with_valid_citations": (
            valid_citation_records
        ),
        "passed": passed,
        "errors": errors,
        "empty_answers": empty_answers,
        "no_citations": no_citations,
        "unsupported_citations": unsupported,
        "unsupported_citation_count": (
            unsupported_citation_count
        ),
        "citation_presence_rate": (
            citation_present / answers_available
            if answers_available
            else 0.0
        ),
        "citation_validity_rate": (
            valid_citation_records / citation_present
            if citation_present
            else 0.0
        ),
        "overall_pass_rate": (
            passed / total
            if total
            else 0.0
        ),
        "answer_success_rate": (
            answers_available / total
            if total
            else 0.0
        ),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    records = load_jsonl(
        INPUT_FILE
    )

    print(
        f"Loaded {len(records)} answer records"
    )

    results = [
        evaluate_record(record)
        for record in records
    ]

    summary = calculate_summary(
        results
    )

    output = {
        "input_file": str(INPUT_FILE),
        "summary": summary,
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("CITATION EVALUATION")
    print("=" * 70)

    print(
        f"Total records          : "
        f"{summary['total_records']}"
    )

    print(
        f"Answers available      : "
        f"{summary['answers_available']}"
    )

    print(
        f"Records with citations : "
        f"{summary['records_with_citations']}"
    )

    print(
        f"Valid citation records : "
        f"{summary['records_with_valid_citations']}"
    )

    print(
        f"Unsupported citations  : "
        f"{summary['unsupported_citations']}"
    )

    print(
        f"Empty answers          : "
        f"{summary['empty_answers']}"
    )

    print(
        f"API/errors             : "
        f"{summary['errors']}"
    )

    print("-" * 70)

    print(
        f"Citation presence rate : "
        f"{summary['citation_presence_rate']:.4f}"
    )

    print(
        f"Citation validity rate : "
        f"{summary['citation_validity_rate']:.4f}"
    )

    print(
        f"Answer success rate    : "
        f"{summary['answer_success_rate']:.4f}"
    )

    print(
        f"Overall pass rate      : "
        f"{summary['overall_pass_rate']:.4f}"
    )

    print("=" * 70)

    print(
        f"Results written to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()