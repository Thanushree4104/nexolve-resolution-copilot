import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CitationValidationResult:
    passed: bool
    citations: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    uncited_sections: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


class CitationValidator:
    """
    Validates citations in RAG-generated answers.

    Expected citation format:
        [KB-1001]

    Also accepts common LLM variants such as:
        (KB-1001)
        KB-1001
        [KB-1001, KB-1002]

    The validator checks:
    1. Citation IDs exist in the retrieved articles.
    2. Required answer sections contain citations when they contain
       KB-derived factual claims.
    3. Unsupported KB IDs are rejected.
    4. Citation validation is compatible with the existing RAGAnswerService.
    """

    REQUIRED_SECTIONS = [
        "Likely issue:",
        "Recommended troubleshooting steps:",
        "Escalation condition:",
        "Agent response:",
    ]

    # Citation formats accepted from LLM output.
    CITATION_PATTERN = re.compile(
        r"""
        (?:
            \[
                \s*
                (KB-\d+)
                (?:\s*,\s*KB-\d+)*
                \s*
            \]
        )
        |
        (?:
            \(
                \s*
                (KB-\d+)
                \s*
            \)
        )
        |
        (?<![A-Za-z0-9_-])
        (KB-\d+)
        (?![A-Za-z0-9_-])
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    SECTION_PATTERN = re.compile(
        r"^(Likely issue|Recommended troubleshooting steps|"
        r"Escalation condition|Agent response):\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(self, require_section_citations=True):
        self.require_section_citations = require_section_citations

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def validate(
        self,
        answer: str,
        retrieved_articles: list[dict[str, Any]] | None = None,
        articles: list[dict[str, Any]] | None = None,
    ) -> CitationValidationResult:

        # Support both names so old/new callers work.
        if retrieved_articles is None:
            retrieved_articles = articles or []

        answer = str(answer or "").strip()

        if not answer:
            return CitationValidationResult(
                passed=False,
                violations=["empty_answer"],
            )

        allowed_ids = self._get_allowed_ids(retrieved_articles)

        citations = self._extract_citations(answer)

        unsupported_citations = [
            citation
            for citation in citations
            if citation.upper() not in allowed_ids
        ]

        violations = []

        # -----------------------------------------------------
        # Unsupported citations
        # -----------------------------------------------------

        for citation in unsupported_citations:
            violations.append(
                f"unsupported_citation:{citation}"
            )

        # -----------------------------------------------------
        # Required section checks
        # -----------------------------------------------------

        sections = self._split_sections(answer)

        if self.require_section_citations:
            for section_name in [
                "Likely issue:",
                "Recommended troubleshooting steps:",
                "Escalation condition:",
            ]:
                content = sections.get(
                    section_name,
                    "",
                ).strip()

                if not content:
                    continue

                if not self._contains_citation(content):
                    violations.append(
                        f"missing_citation:{section_name}"
                    )

        # -----------------------------------------------------
        # Detect uncited factual lines
        # -----------------------------------------------------

        unsupported_claims = []

        for section_name in [
            "Likely issue:",
            "Recommended troubleshooting steps:",
            "Escalation condition:",
        ]:
            content = sections.get(
                section_name,
                "",
            ).strip()

            if not content:
                continue

            lines = self._meaningful_lines(content)

            for line in lines:

                # A line that explicitly says the KB is insufficient
                # is not treated as a KB-derived factual claim.
                if self._is_insufficient_evidence_statement(line):
                    continue

                # Pure numbering/bullets are still claims if they
                # contain actual text.
                if self._contains_citation(line):
                    continue

                # Ignore very short structural fragments.
                if self._is_structural_line(line):
                    continue

                unsupported_claims.append(
                    self._clean_claim(line)
                )

        for claim in unsupported_claims:
            violations.append(
                f"uncited_claim:{claim}"
            )

        # -----------------------------------------------------
        # Determine final status
        # -----------------------------------------------------

        passed = len(violations) == 0

        return CitationValidationResult(
            passed=passed,
            citations=citations,
            unsupported_citations=unsupported_citations,
            uncited_sections=self._uncited_sections(
                sections
            ),
            unsupported_claims=unsupported_claims,
            violations=violations,
        )

    # ---------------------------------------------------------
    # Allowed KB IDs
    # ---------------------------------------------------------

    def _get_allowed_ids(
        self,
        retrieved_articles: list[dict[str, Any]],
    ) -> set[str]:

        allowed = set()

        for article in retrieved_articles or []:
            if not isinstance(article, dict):
                continue

            article_id = (
                article.get("id")
                or article.get("article_id")
                or article.get("kb_id")
            )

            if article_id:
                allowed.add(
                    str(article_id)
                    .strip()
                    .upper()
                )

        return allowed

    # ---------------------------------------------------------
    # Citation extraction
    # ---------------------------------------------------------

    def _extract_citations(
        self,
        text: str,
    ) -> list[str]:

        found = []

        for match in self.CITATION_PATTERN.finditer(text):

            # Because the regex has multiple groups, take
            # whichever group matched.
            groups = match.groups()

            for group in groups:
                if group:
                    citation = group.upper()

                    if citation not in found:
                        found.append(citation)

                    break

        return found

    def _contains_citation(
        self,
        text: str,
    ) -> bool:

        return bool(
            self.CITATION_PATTERN.search(
                text
            )
        )

    # ---------------------------------------------------------
    # Section parsing
    # ---------------------------------------------------------

    def _split_sections(
        self,
        answer: str,
    ) -> dict[str, str]:

        sections = {}

        matches = list(
            self.SECTION_PATTERN.finditer(
                answer
            )
        )

        if not matches:
            return sections

        for index, match in enumerate(matches):

            section_name = (
                match.group(1).strip()
                + ":"
            )

            start = match.end()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(answer)

            content = answer[
                start:end
            ].strip()

            sections[section_name] = content

        return sections

    # ---------------------------------------------------------
    # Claim helpers
    # ---------------------------------------------------------

    def _meaningful_lines(
        self,
        content: str,
    ) -> list[str]:

        lines = []

        for raw_line in content.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            # Remove markdown bullets/numbering for analysis.
            cleaned = re.sub(
                r"^\s*(?:[-*•]|\d+[.)])\s*",
                "",
                line,
            ).strip()

            if cleaned:
                lines.append(cleaned)

        return lines

    def _is_insufficient_evidence_statement(
        self,
        line: str,
    ) -> bool:

        normalized = line.lower()

        phrases = [
            "available knowledge-base evidence is insufficient",
            "available knowledge base evidence is insufficient",
            "knowledge-base evidence is insufficient",
            "knowledge base evidence is insufficient",
            "not enough information",
            "insufficient information",
            "cannot be determined from the available",
            "cannot determine from the available",
        ]

        return any(
            phrase in normalized
            for phrase in phrases
        )

    def _is_structural_line(
        self,
        line: str,
    ) -> bool:

        normalized = line.strip().lower()

        if normalized in {
            "none",
            "none provided",
            "n/a",
            "not applicable",
        }:
            return True

        return False

    def _clean_claim(
        self,
        claim: str,
    ) -> str:

        claim = re.sub(
            r"\s+",
            " ",
            claim,
        )

        return claim.strip()

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    def _uncited_sections(
        self,
        sections: dict[str, str],
    ) -> list[str]:

        result = []

        for section_name in [
            "Likely issue:",
            "Recommended troubleshooting steps:",
            "Escalation condition:",
        ]:
            content = sections.get(
                section_name,
                "",
            ).strip()

            if (
                content
                and not self._contains_citation(content)
            ):
                result.append(section_name)

        return result