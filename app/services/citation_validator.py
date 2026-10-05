import re
from dataclasses import dataclass, field
from typing import Any


REQUIRED_SECTIONS = [
    "Likely issue:",
    "Recommended troubleshooting steps:",
    "Escalation condition:",
    "Agent response:",
]


@dataclass
class CitationValidationResult:
    passed: bool

    citations: list[str] = field(
        default_factory=list
    )

    unsupported_citations: list[str] = field(
        default_factory=list
    )

    uncited_sections: list[str] = field(
        default_factory=list
    )

    unsupported_claims: list[str] = field(
        default_factory=list
    )

    violations: list[str] = field(
        default_factory=list
    )


class CitationValidator:
    """
    Lenient but safe citation validator for RAG answers.

    The validator checks:

    1. Required logical sections exist.
    2. KB citations refer to retrieved articles.
    3. Factual claims in the main KB-derived sections have citations.
    4. Markdown formatting around headings is tolerated.
    5. Wrapped lines are treated as part of the same claim.
    6. Insufficient-evidence statements do not require citations.

    Accepted citation formats:

        [KB-1028]
        (KB-1028)
        KB-1028
        [KB-1028, KB-1029]
    """

    # =========================================================
    # SECTION DEFINITIONS
    # =========================================================

    REQUIRED_SECTIONS = [
        "Likely issue:",
        "Recommended troubleshooting steps:",
        "Escalation condition:",
        "Agent response:",
    ]

    # More tolerant section matcher.
    #
    # Accepts:
    #
    # Likely issue:
    # Likely issue :
    # ## Likely issue:
    # **Likely issue:**
    # ### Recommended troubleshooting steps:
    #
    SECTION_HEADER_PATTERN = re.compile(
        r"""
        ^\s*
        (?:[#>*\-\s]*)?
        (?:\*\*)?
        (?P<section>
            Likely\s+issue
            |
            Recommended\s+troubleshooting\s+steps
            |
            Escalation\s+condition
            |
            Agent\s+response
        )
        \s*
        :?
        \s*
        (?:\*\*)?
        \s*$
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )

    # =========================================================
    # CITATION PATTERN
    # =========================================================

    CITATION_PATTERN = re.compile(
        r"""
        \[
            \s*
            (KB-\d+)
            (?:
                \s*,\s*
                (KB-\d+)
            )*
            \s*
        \]
        |
        \(
            \s*
            (KB-\d+)
            \s*
        \)
        |
        (?<![A-Za-z0-9_-])
        (KB-\d+)
        (?![A-Za-z0-9_-])
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        require_section_citations=False,
    ):
        """
        Section-level citation requirements are intentionally
        disabled.

        We validate citations at claim level instead.

        This prevents harmless responses from failing simply
        because one section contains a general explanation.
        """

        self.require_section_citations = (
            require_section_citations
        )

    # =========================================================
    # PUBLIC VALIDATION
    # =========================================================

    def validate(
        self,
        answer: str,
        retrieved_articles: list[dict[str, Any]] | None = None,
        articles: list[dict[str, Any]] | None = None,
    ) -> CitationValidationResult:

        if retrieved_articles is None:
            retrieved_articles = articles or []

        answer = str(
            answer or ""
        ).strip()

        if not answer:

            return CitationValidationResult(
                passed=False,
                violations=[
                    "empty_answer"
                ],
            )

        violations = []

        # -----------------------------------------------------
        # ALLOWED KB IDS
        # -----------------------------------------------------

        allowed_ids = self._get_allowed_ids(
            retrieved_articles
        )

        # -----------------------------------------------------
        # EXTRACT CITATIONS
        # -----------------------------------------------------

        citations = self._extract_citations(
            answer
        )

        # -----------------------------------------------------
        # UNSUPPORTED CITATIONS
        # -----------------------------------------------------

        unsupported_citations = [
            citation
            for citation in citations
            if citation.upper() not in allowed_ids
        ]

        for citation in unsupported_citations:

            violations.append(
                f"unsupported_citation:{citation}"
            )

        # -----------------------------------------------------
        # SPLIT INTO SECTIONS
        # -----------------------------------------------------

        sections = self._split_sections(
            answer
        )

        # -----------------------------------------------------
        # REQUIRED SECTION CHECK
        # -----------------------------------------------------

        missing_sections = []

        normalized_section_names = {
            self._normalize_section_name(name)
            for name in sections
        }

        for required in self.REQUIRED_SECTIONS:

            normalized_required = (
                self._normalize_section_name(
                    required
                )
            )

            if (
                normalized_required
                not in normalized_section_names
            ):

                missing_sections.append(
                    required
                )

                violations.append(
                    f"missing_section:{required}"
                )

        # -----------------------------------------------------
        # CLAIM VALIDATION
        # -----------------------------------------------------

        unsupported_claims = []

        # Agent response is intentionally excluded from
        # strict claim-level citation checking.
        #
        # It is customer-facing conversational text and often
        # repeats already-cited troubleshooting instructions.
        #
        # The KB-derived sections remain citation-validated.

        sections_to_validate = [
            "Likely issue:",
            "Recommended troubleshooting steps:",
            "Escalation condition:",
        ]

        for section_name in sections_to_validate:

            content = self._get_section(
                sections,
                section_name,
            )

            if not content:
                continue

            claims = self._meaningful_lines(
                content
            )

            for claim in claims:

                # Empty / structural content.
                if not claim:
                    continue

                # Evidence-insufficient statements do not
                # require citations.
                if self._is_insufficient_evidence_statement(
                    claim
                ):
                    continue

                # Structural values don't need citations.
                if self._is_structural_line(
                    claim
                ):
                    continue

                # Citation exists → accepted.
                if self._contains_citation(
                    claim
                ):
                    continue

                unsupported_claims.append(
                    self._clean_claim(
                        claim
                    )
                )

        # -----------------------------------------------------
        # ADD CLAIM VIOLATIONS
        # -----------------------------------------------------

        for claim in unsupported_claims:

            violations.append(
                f"uncited_claim:{claim}"
            )

        # -----------------------------------------------------
        # FINAL RESULT
        # -----------------------------------------------------

        return CitationValidationResult(
            passed=len(violations) == 0,

            citations=citations,

            unsupported_citations=(
                unsupported_citations
            ),

            uncited_sections=(
                self._uncited_sections(
                    sections
                )
            ),

            unsupported_claims=(
                unsupported_claims
            ),

            violations=violations,
        )

    # =========================================================
    # ALLOWED IDS
    # =========================================================

    def _get_allowed_ids(
        self,
        retrieved_articles: list[dict[str, Any]],
    ) -> set[str]:

        allowed = set()

        for article in (
            retrieved_articles or []
        ):

            if not isinstance(
                article,
                dict,
            ):
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

    # =========================================================
    # CITATION EXTRACTION
    # =========================================================

    def _extract_citations(
        self,
        text: str,
    ) -> list[str]:

        found = []

        for match in self.CITATION_PATTERN.finditer(
            text
        ):

            for group in match.groups():

                if not group:
                    continue

                citation = group.upper()

                if citation not in found:
                    found.append(
                        citation
                    )

                break

        return found

    # =========================================================
    # CITATION CHECK
    # =========================================================

    def _contains_citation(
        self,
        text: str,
    ) -> bool:

        return bool(
            self.CITATION_PATTERN.search(
                text
            )
        )

    # =========================================================
    # SECTION PARSING
    # =========================================================

    def _split_sections(
        self,
        answer: str,
    ) -> dict[str, str]:

        sections = {}

        matches = list(
            self.SECTION_HEADER_PATTERN.finditer(
                answer
            )
        )

        if not matches:
            return sections

        for index, match in enumerate(
            matches
        ):

            raw_name = match.group(
                "section"
            )

            section_name = (
                self._canonical_section_name(
                    raw_name
                )
            )

            start = match.end()

            if index + 1 < len(matches):

                end = matches[
                    index + 1
                ].start()

            else:

                end = len(answer)

            content = answer[
                start:end
            ].strip()

            sections[
                section_name
            ] = content

        return sections

    # =========================================================
    # SECTION HELPERS
    # =========================================================

    def _canonical_section_name(
        self,
        name: str,
    ) -> str:

        normalized = (
            re.sub(
                r"\s+",
                " ",
                name.strip().lower(),
            )
        )

        mapping = {
            "likely issue":
                "Likely issue:",

            "recommended troubleshooting steps":
                "Recommended troubleshooting steps:",

            "escalation condition":
                "Escalation condition:",

            "agent response":
                "Agent response:",
        }

        return mapping.get(
            normalized,
            name.strip() + ":",
        )

    def _normalize_section_name(
        self,
        name: str,
    ) -> str:

        return re.sub(
            r"[^a-z0-9]+",
            " ",
            name.lower(),
        ).strip()

    def _get_section(
        self,
        sections: dict[str, str],
        section_name: str,
    ) -> str:

        target = self._normalize_section_name(
            section_name
        )

        for name, content in sections.items():

            if (
                self._normalize_section_name(
                    name
                )
                == target
            ):

                return content

        return ""

    # =========================================================
    # CLAIM SPLITTING
    # =========================================================

    def _meaningful_lines(
        self,
        content: str,
    ) -> list[str]:

        claims = []

        current = ""

        for raw_line in content.splitlines():

            line = raw_line.strip()

            if not line:

                if current:
                    claims.append(
                        current.strip()
                    )

                    current = ""

                continue

            # Detect bullets / numbered items.
            is_new_item = bool(
                re.match(
                    r"^\s*(?:[-*•]|\d+[.)])\s+",
                    line,
                )
            )

            cleaned = re.sub(
                r"^\s*(?:[-*•]|\d+[.)])\s*",
                "",
                line,
            ).strip()

            if not cleaned:
                continue

            if is_new_item:

                if current:
                    claims.append(
                        current.strip()
                    )

                current = cleaned

            else:

                if current:

                    current += (
                        " "
                        + cleaned
                    )

                else:

                    current = cleaned

        if current:
            claims.append(
                current.strip()
            )

        return claims

    # =========================================================
    # INSUFFICIENT EVIDENCE
    # =========================================================

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
            "cannot be determined using the available",
            "cannot determine using the available",
            "not specified in the knowledge base",
            "not provided in the knowledge base",
            "the knowledge base does not specify",
        ]

        return any(
            phrase in normalized
            for phrase in phrases
        )

    # =========================================================
    # STRUCTURAL LINES
    # =========================================================

    def _is_structural_line(
        self,
        line: str,
    ) -> bool:

        normalized = (
            line.strip().lower()
        )

        return normalized in {
            "",
            "none",
            "none provided",
            "n/a",
            "not applicable",
            "-",
            "—",
            "no information available",
        }

    # =========================================================
    # CLAIM CLEANING
    # =========================================================

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

    # =========================================================
    # DIAGNOSTICS
    # =========================================================

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

            content = self._get_section(
                sections,
                section_name,
            ).strip()

            if not content:
                continue

            if self._is_insufficient_evidence_statement(
                content
            ):
                continue

            if not self._contains_citation(
                content
            ):
                result.append(
                    section_name
                )

        return result