import json
import os
import time
from pathlib import Path

from app.llm.factory import get_provider
from app.services.citation_validator import CitationValidator
from app.services.rag_answer import RAGAnswerService
from app.services.retriever import HybridRetriever


EVAL_PATH = Path(
    "data/eval/retrieval_eval.jsonl"
)

KB_PATH = Path(
    "data/synthetic/kb_articles.jsonl"
)

OUTPUT_PATH = Path(
    "data/eval/answer_results.jsonl"
)

MODEL = "intfloat/multilingual-e5-small"


def load_jsonl(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def get_eval_limit():
    value = os.getenv(
        "EVAL_LIMIT",
        "",
    ).strip()

    if not value:
        return None

    try:
        limit = int(value)

        if limit <= 0:
            return None

        return limit

    except ValueError:
        print(
            f"WARNING: Invalid EVAL_LIMIT={value!r}. "
            "Using all queries."
        )
        return None


def get_provider_name():
    return os.getenv(
        "LLM_PROVIDER",
        "mock",
    ).lower().strip()


def validate_provider_configuration():
    provider = get_provider_name()

    if provider != "groq":
        raise RuntimeError(
            "Answer evaluation must use Groq.\n\n"
            f"Current LLM_PROVIDER={provider!r}\n\n"
            "Set it before running evaluation:\n"
            '  $env:LLM_PROVIDER="groq"\n\n'
            "Then run:\n"
            "  python -m scripts.eval_answers"
        )


def get_error_type(error):
    text = str(error).lower()

    if (
        "rate limit" in text
        or "429" in text
        or "too many requests" in text
    ):
        return "rate_limit"

    if (
        "unavailable" in text
        or "connection" in text
        or "timeout" in text
    ):
        return "unavailable"

    return "other"


def count_citations(answer):
    if not answer:
        return 0

    import re

    matches = re.findall(
        r"\[KB-[A-Za-z0-9_-]+\]",
        answer,
    )

    return len(matches)


def get_retrieved_ids(
    retriever,
    query,
    top_k=5,
):
    """
    Retrieve documents using the production retriever.

    search_all() is used when available because the
    evaluation needs dense/BM25/hybrid rankings.
    """

    result = retriever.search_all(
        query,
        top_k=top_k,
    )

    hybrid = result.get(
        "hybrid",
        [],
    )

    return hybrid


def main():
    validate_provider_configuration()

    evaluation = load_jsonl(
        EVAL_PATH
    )

    kb_articles = load_jsonl(
        KB_PATH
    )

    eval_limit = get_eval_limit()

    if eval_limit is not None:
        evaluation = evaluation[
            :eval_limit
        ]

    provider_name = get_provider_name()

    print(
        "=" * 70
    )
    print(
        "ANSWER EVALUATION CONFIGURATION"
    )
    print(
        "=" * 70
    )

    print(
        f"Provider                : "
        f"{provider_name}"
    )

    print(
        f"Evaluation limit        : "
        f"{eval_limit if eval_limit else 'ALL'}"
    )

    print(
        f"Evaluation queries      : "
        f"{len(evaluation)}"
    )

    print(
        f"Evaluation file         : "
        f"{EVAL_PATH}"
    )

    print(
        f"Output file             : "
        f"{OUTPUT_PATH}"
    )

    print(
        "Mode                    : GROQ / REAL LLM"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Initializing shared "
        "HybridRetriever..."
    )

    retriever = HybridRetriever(
        kb_path=str(KB_PATH),
        model_name=MODEL,
    )

    print(
        "Initializing provider: groq"
    )

    provider = get_provider()

    rag_service = RAGAnswerService(
        provider=provider,
        retriever=retriever,
        max_articles=2,
    )

    # Citation validator used by the RAG service.
    citation_validator = CitationValidator()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Start a fresh output file for this evaluation.
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        attempted = 0
        answers_generated = 0
        successful_pipeline = 0

        guardrail_failures = 0
        citation_failures = 0
        rate_limit_failures = 0
        llm_unavailable = 0
        llm_empty = 0
        retrieval_failures = 0
        other_errors = 0

        total_answer_words = 0
        total_retrieval_latency = 0.0
        total_answer_latency = 0.0

        for position, item in enumerate(
            evaluation,
            start=1,
        ):
            attempted += 1

            ticket_id = item[
                "ticket_id"
            ]

            complaint = item[
                "query"
            ]

            expected_ids = item.get(
                "relevant_ids",
                [],
            )

            print(
                f"[{position}/{len(evaluation)}] "
                f"{ticket_id}"
            )

            result = {
                "ticket_id": ticket_id,
                "query": complaint,
                "relevant_ids": expected_ids,
                "retrieved_ids": [],
                "retrieval_hit@1": False,
                "retrieval_hit@3": False,
                "retrieval_rr": 0.0,
                "answer": "",
                "answer_available": False,
                "answer_word_count": 0,
                "citation_count": 0,
                "error": None,
                "status": "unknown",
                "retrieval_latency_ms": 0.0,
                "answer_latency_ms": 0.0,
                "latency_ms": 0.0,
            }

            overall_start = time.perf_counter()

            # -------------------------------------------------
            # RETRIEVAL
            # -------------------------------------------------

            retrieval_start = time.perf_counter()

            try:
                retrieved_ids = get_retrieved_ids(
                    retriever,
                    complaint,
                    top_k=len(kb_articles),
                )

                result[
                    "retrieved_ids"
                ] = retrieved_ids

                if expected_ids:
                    result[
                        "retrieval_hit@1"
                    ] = bool(
                        set(
                            retrieved_ids[:1]
                        )
                        & set(expected_ids)
                    )

                    result[
                        "retrieval_hit@3"
                    ] = bool(
                        set(
                            retrieved_ids[:3]
                        )
                        & set(expected_ids)
                    )

                    rr = 0.0

                    for rank, kb_id in enumerate(
                        retrieved_ids,
                        start=1,
                    ):
                        if kb_id in expected_ids:
                            rr = 1.0 / rank
                            break

                    result[
                        "retrieval_rr"
                    ] = rr

            except Exception as error:
                retrieval_failures += 1

                result[
                    "status"
                ] = "retrieval_failure"

                result[
                    "error"
                ] = (
                    "Retrieval failed: "
                    + str(error)
                )

                result[
                    "retrieval_latency_ms"
                ] = round(
                    (
                        time.perf_counter()
                        - retrieval_start
                    )
                    * 1000,
                    1,
                )

                result[
                    "latency_ms"
                ] = round(
                    (
                        time.perf_counter()
                        - overall_start
                    )
                    * 1000,
                    1,
                )

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                continue

            result[
                "retrieval_latency_ms"
            ] = round(
                (
                    time.perf_counter()
                    - retrieval_start
                )
                * 1000,
                1,
            )

            total_retrieval_latency += (
                result[
                    "retrieval_latency_ms"
                ]
            )

            # -------------------------------------------------
            # ANSWER GENERATION
            # -------------------------------------------------

            answer_start = time.perf_counter()

            try:
                answer = rag_service.answer(
                    complaint
                )

                if answer is None:
                    answer = ""

                answer = str(
                    answer
                ).strip()

                result[
                    "answer"
                ] = answer

                result[
                    "answer_available"
                ] = bool(answer)

                result[
                    "answer_word_count"
                ] = len(
                    answer.split()
                )

                result[
                    "citation_count"
                ] = count_citations(
                    answer
                )

                if not answer:
                    llm_empty += 1

                    result[
                        "status"
                    ] = "llm_empty"

                    result[
                        "error"
                    ] = (
                        "LLM returned an empty "
                        "answer."
                    )

                else:
                    answers_generated += 1

                    total_answer_words += (
                        result[
                            "answer_word_count"
                        ]
                    )

                    # -----------------------------------------
                    # CITATION VALIDATION
                    # -----------------------------------------

                    try:
                        validation = (
                            citation_validator.validate(
                                answer
                            )
                        )

                        # Different validator implementations
                        # may return True/False or an object.
                        valid = bool(
                            validation
                        )

                        if not valid:
                            citation_failures += 1

                            result[
                                "status"
                            ] = (
                                "citation_failure"
                            )

                            result[
                                "error"
                            ] = (
                                "Generated answer "
                                "failed citation "
                                "validation."
                            )

                        else:
                            successful_pipeline += 1

                            result[
                                "status"
                            ] = "success"

                    except Exception as validation_error:
                        citation_failures += 1

                        result[
                            "status"
                        ] = (
                            "citation_failure"
                        )

                        result[
                            "error"
                        ] = (
                            "Generated answer "
                            "failed citation "
                            "validation: "
                            + str(
                                validation_error
                            )
                        )

            except Exception as error:
                error_type = get_error_type(
                    error
                )

                if error_type == "rate_limit":
                    rate_limit_failures += 1

                    result[
                        "status"
                    ] = "rate_limit"

                elif error_type == "unavailable":
                    llm_unavailable += 1

                    result[
                        "status"
                    ] = "llm_unavailable"

                else:
                    other_errors += 1

                    result[
                        "status"
                    ] = "error"

                result[
                    "error"
                ] = str(error)

            result[
                "answer_latency_ms"
            ] = round(
                (
                    time.perf_counter()
                    - answer_start
                )
                * 1000,
                1,
            )

            total_answer_latency += (
                result[
                    "answer_latency_ms"
                ]
            )

            result[
                "latency_ms"
            ] = round(
                (
                    time.perf_counter()
                    - overall_start
                )
                * 1000,
                1,
            )

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    query_count = len(
        evaluation
    )

    mean_answer_words = (
        total_answer_words
        / answers_generated
        if answers_generated
        else 0.0
    )

    mean_retrieval_latency = (
        total_retrieval_latency
        / attempted
        if attempted
        else 0.0
    )

    mean_answer_latency = (
        total_answer_latency
        / attempted
        if attempted
        else 0.0
    )

    citation_presence_rate = 0.0

    if answers_generated:
        citation_presence_rate = (
            sum(
                1
                for line in OUTPUT_PATH.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
                and json.loads(line).get(
                    "citation_count",
                    0,
                )
                > 0
            )
            / answers_generated
        )

    retrieval_hit_1 = []
    retrieval_hit_3 = []
    retrieval_rr = []

    for line in OUTPUT_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        item = json.loads(line)

        retrieval_hit_1.append(
            float(
                item.get(
                    "retrieval_hit@1",
                    False,
                )
            )
        )

        retrieval_hit_3.append(
            float(
                item.get(
                    "retrieval_hit@3",
                    False,
                )
            )
        )

        retrieval_rr.append(
            float(
                item.get(
                    "retrieval_rr",
                    0.0,
                )
            )
        )

    print()
    print(
        "=" * 70
    )
    print(
        "ANSWER EVALUATION"
    )
    print(
        "=" * 70
    )

    print(
        f"Provider                : "
        f"{provider_name}"
    )

    print(
        f"Queries attempted       : "
        f"{attempted}"
    )

    print(
        f"Answers generated       : "
        f"{answers_generated}"
    )

    print(
        f"Successful pipeline     : "
        f"{successful_pipeline}"
    )

    print(
        f"Guardrail failures      : "
        f"{guardrail_failures}"
    )

    print(
        f"Citation failures      : "
        f"{citation_failures}"
    )

    print(
        f"LLM rate-limit failures : "
        f"{rate_limit_failures}"
    )

    print(
        f"LLM unavailable         : "
        f"{llm_unavailable}"
    )

    print(
        f"LLM empty responses     : "
        f"{llm_empty}"
    )

    print(
        f"Retrieval failures      : "
        f"{retrieval_failures}"
    )

    print(
        f"Other errors            : "
        f"{other_errors}"
    )

    print(
        "-" * 70
    )

    if retrieval_hit_1:
        print(
            f"Retrieval Hit@1         : "
            f"{np_mean(retrieval_hit_1):.4f}"
        )

    if retrieval_hit_3:
        print(
            f"Retrieval Hit@3         : "
            f"{np_mean(retrieval_hit_3):.4f}"
        )

    if retrieval_rr:
        print(
            f"Retrieval MRR           : "
            f"{np_mean(retrieval_rr):.4f}"
        )

    print(
        f"Mean answer words       : "
        f"{mean_answer_words:.1f}"
    )

    print(
        f"Citation presence rate  : "
        f"{citation_presence_rate:.4f}"
    )

    print(
        f"Mean retrieval latency  : "
        f"{mean_retrieval_latency:.1f} ms"
    )

    print(
        f"Mean answer latency     : "
        f"{mean_answer_latency:.1f} ms"
    )

    print(
        "=" * 70
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )


def np_mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


if __name__ == "__main__":
    main()