import re
from dataclasses import dataclass
from typing import List


REQUIRED_SECTIONS = [
    "Likely issue:",
    "Recommended troubleshooting steps:",
    "Escalation condition:",
    "Agent response:",
]


@dataclass
class GuardrailResult:
    passed: bool
    answer: str
    violations: List[str]


class OutputGuardrail:
    """
    Validates an LLM-generated RAG answer against:

    1. Required response structure
    2. Retrieved KB references
    3. Forbidden action/commitment claims

    The guardrail does NOT perform citation validation.
    Citation validation is handled by CitationValidator.
    """

    # ---------------------------------------------------------
    # Claims that imply an action has already happened
    # or that the agent is promising to perform something.
    # ---------------------------------------------------------

    FORBIDDEN_PATTERNS = [
        # Already performed actions
        r"\bI have checked\b",
        r"\bwe have checked\b",
        r"\bI checked\b",
        r"\bwe checked\b",
        r"\bhas been checked\b",

        r"\bI have fixed\b",
        r"\bwe have fixed\b",
        r"\bI fixed\b",
        r"\bwe fixed\b",
        r"\bhas been fixed\b",

        r"\bI have resolved\b",
        r"\bwe have resolved\b",
        r"\bI resolved\b",
        r"\bwe resolved\b",
        r"\bhas been resolved\b",

        # Future commitments
        r"\bI will\b",
        r"\bwe will\b",
        r"\bI['’]ll\b",
        r"\bwe['’]ll\b",

        r"\bI am going to\b",
        r"\bwe are going to\b",
        r"\bI'm going to\b",
        r"\bwe're going to\b",

        # Actions currently being performed
        r"\bI am checking\b",
        r"\bwe are checking\b",

        r"\bI am investigating\b",
        r"\bwe are investigating\b",

        r"\bI am performing\b",
        r"\bwe are performing\b",

        r"\bI am fixing\b",
        r"\bwe are fixing\b",

        # Explicit promises
        r"\bI promise\b",
        r"\bwe promise\b",

        r"\bI['’]ll keep you updated\b",
        r"\bwe['’]ll keep you updated\b",

        r"\bI['’]ll let you know\b",
        r"\bwe['’]ll let you know\b",
    ]

    # ---------------------------------------------------------
    # KB citation/reference pattern.
    # ---------------------------------------------------------

    KB_PATTERN = re.compile(
        r"\bKB-\d+\b",
        re.IGNORECASE,
    )

    def validate(
        self,
        answer: str,
        retrieved_articles: list,
    ) -> GuardrailResult:

        violations = []

        answer = str(answer or "").strip()

        # =====================================================
        # EMPTY OUTPUT
        # =====================================================

        if not answer:
            return GuardrailResult(
                passed=False,
                answer=answer,
                violations=[
                    "empty_llm_output"
                ],
            )

        # =====================================================
        # REQUIRED STRUCTURE
        # =====================================================

        for section in REQUIRED_SECTIONS:

            if section not in answer:
                violations.append(
                    f"missing_section:{section}"
                )

        # =====================================================
        # RETRIEVED KB IDS
        # =====================================================

        retrieved_ids = set()

        for article in retrieved_articles or []:

            if not isinstance(article, dict):
                continue

            article_id = (
                article.get("id")
                or article.get("article_id")
                or article.get("kb_id")
            )

            if article_id:
                retrieved_ids.add(
                    str(article_id)
                    .strip()
                    .upper()
                )

        # =====================================================
        # CHECK KB REFERENCES
        # =====================================================

        referenced_ids = {
            match.upper()
            for match in self.KB_PATTERN.findall(answer)
        }

        unknown_ids = referenced_ids - retrieved_ids

        for kb_id in sorted(unknown_ids):

            violations.append(
                f"unsupported_kb_reference:{kb_id}"
            )

        # =====================================================
        # FORBIDDEN ACTION CLAIMS
        # =====================================================

        for pattern in self.FORBIDDEN_PATTERNS:

            if re.search(
                pattern,
                answer,
                flags=re.IGNORECASE,
            ):

                violations.append(
                    f"forbidden_action_claim:{pattern}"
                )

        # =====================================================
        # REMOVE DUPLICATES
        # =====================================================

        violations = list(
            dict.fromkeys(violations)
        )

        # =====================================================
        # RESULT
        # =====================================================

        return GuardrailResult(
            passed=len(violations) == 0,
            answer=answer,
            violations=violations,
        )