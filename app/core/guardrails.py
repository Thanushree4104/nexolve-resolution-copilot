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
    Validates an LLM-generated RAG answer against
    structural and evidence-grounding constraints.
    """

    FORBIDDEN_PATTERNS = [
        r"\bI have checked\b",
        r"\bwe have checked\b",
        r"\bI checked\b",
        r"\bwe checked\b",
        r"\bhas been checked\b",
        r"\bhas been fixed\b",
        r"\bhas been resolved\b",
        r"\bwe will fix\b",
        r"\bwe will investigate\b",
        r"\bwe will perform\b",
        r"\bI will investigate\b",
        r"\bI will perform\b",
    ]

    def validate(
        self,
        answer: str,
        retrieved_articles: list,
    ) -> GuardrailResult:

        violations = []

        if not answer or not answer.strip():
            violations.append(
                "empty_llm_output"
            )

            return GuardrailResult(
                passed=False,
                answer=answer,
                violations=violations,
            )

        # --------------------------------------------------
        # STRUCTURE VALIDATION
        # --------------------------------------------------

        for section in REQUIRED_SECTIONS:
            if section not in answer:
                violations.append(
                    f"missing_section:{section}"
                )

        # --------------------------------------------------
        # RETRIEVED KB IDS
        # --------------------------------------------------

        retrieved_ids = {
            article["id"]
            for article in retrieved_articles
        }

        referenced_ids = set(
            re.findall(
                r"\bKB-\d+\b",
                answer,
            )
        )

        unknown_ids = (
            referenced_ids
            - retrieved_ids
        )

        for kb_id in sorted(unknown_ids):
            violations.append(
                f"unsupported_kb_reference:{kb_id}"
            )

        # --------------------------------------------------
        # FORBIDDEN ACTION CLAIMS
        # --------------------------------------------------

        answer_lower = answer.lower()

        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(
                pattern,
                answer_lower,
                flags=re.IGNORECASE,
            ):
                violations.append(
                    f"forbidden_action_claim:{pattern}"
                )

        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        return GuardrailResult(
            passed=len(violations) == 0,
            answer=answer,
            violations=violations,
        )