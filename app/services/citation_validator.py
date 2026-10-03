import re
from dataclasses import dataclass
from typing import List


CITATION_PATTERN = re.compile(r"\[KB-\d+\]")


@dataclass
class CitationValidationResult:
    passed: bool
    citations: List[str]
    unsupported_citations: List[str]
    uncited_sections: List[str]
    violations: List[str]


class CitationValidator:
    """
    Validates citations in an LLM-generated RAG answer.

    Rules:
    - Citations must use [KB-XXXX] format.
    - A citation must refer to a retrieved KB article.
    - Required factual sections must contain at least one citation.
    """

    REQUIRED_CITED_SECTIONS = [
        "Likely issue:",
        "Recommended troubleshooting steps:",
        "Escalation condition:",
    ]

    def validate(
        self,
        answer: str,
        retrieved_articles: list,
    ) -> CitationValidationResult:

        violations = []

        if not answer or not answer.strip():
            return CitationValidationResult(
                passed=False,
                citations=[],
                unsupported_citations=[],
                uncited_sections=[],
                violations=["empty_answer"],
            )

        retrieved_ids = {
            article["id"]
            for article in retrieved_articles
        }

        citations = sorted(
            set(CITATION_PATTERN.findall(answer))
        )

        citation_ids = {
            citation.strip("[]")
            for citation in citations
        }

        unsupported_citations = sorted(
            citation_ids - retrieved_ids
        )

        for kb_id in unsupported_citations:
            violations.append(
                f"citation_not_retrieved:{kb_id}"
            )

        uncited_sections = []

        for section in self.REQUIRED_CITED_SECTIONS:
            start = answer.find(section)

            if start == -1:
                continue

            remaining = answer[start + len(section):]

            next_section = re.search(
                r"\n(?:Likely issue|Recommended troubleshooting steps|"
                r"Escalation condition|Agent response):",
                remaining,
            )

            if next_section:
                content = remaining[:next_section.start()]
            else:
                content = remaining

            if not CITATION_PATTERN.search(content):
                uncited_sections.append(section)

                violations.append(
                    f"missing_citation:{section}"
                )

        return CitationValidationResult(
            passed=len(violations) == 0,
            citations=citations,
            unsupported_citations=unsupported_citations,
            uncited_sections=uncited_sections,
            violations=violations,
        )