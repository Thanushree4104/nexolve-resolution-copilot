import re
from dataclasses import dataclass
from typing import List, Dict, Set


CITATION_PATTERN = re.compile(r"\[KB-\d+\]")

SECTION_PATTERN = re.compile(
    r"\n(?:Likely issue|Recommended troubleshooting steps|"
    r"Escalation condition|Agent response):",
    re.IGNORECASE,
)


@dataclass
class CitationValidationResult:
    passed: bool
    citations: List[str]
    unsupported_citations: List[str]
    uncited_sections: List[str]
    unsupported_claims: List[str]
    violations: List[str]


class CitationValidator:
    """
    Validates citations in an LLM-generated RAG answer.

    Validation rules:

    1. Citations must use [KB-XXXX] format.
    2. Citations must refer only to retrieved KB articles.
    3. Required factual sections must contain citations.
    4. Claims attached to citations must be supported by the
       cited KB article.
    5. Harmless linguistic differences such as punctuation,
       capitalization, Unicode dashes, and grammatical forms
       should not cause false rejection.
    """

    REQUIRED_CITED_SECTIONS = [
        "Likely issue:",
        "Recommended troubleshooting steps:",
        "Escalation condition:",
    ]

    # Very common words that do not provide useful evidence
    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "can",
        "could",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "then",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "would",
        "customer",
        "issue",
        "problem",
        "case",
        "step",
        "steps",
        "recommended",
        "condition",
    }

    def validate(
        self,
        answer: str,
        retrieved_articles: list,
    ) -> CitationValidationResult:

        violations = []
        unsupported_claims = []

        if not answer or not answer.strip():
            return CitationValidationResult(
                passed=False,
                citations=[],
                unsupported_citations=[],
                uncited_sections=[],
                unsupported_claims=[],
                violations=["empty_answer"],
            )

        # --------------------------------------------------
        # RETRIEVED KB IDS
        # --------------------------------------------------

        retrieved_by_id: Dict[str, dict] = {
            article["id"]: article
            for article in retrieved_articles
            if "id" in article
        }

        retrieved_ids = set(retrieved_by_id.keys())

        # --------------------------------------------------
        # EXTRACT CITATIONS
        # --------------------------------------------------

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

        # --------------------------------------------------
        # REQUIRED SECTION CITATIONS
        # --------------------------------------------------

        uncited_sections = []

        for section in self.REQUIRED_CITED_SECTIONS:

            start = answer.find(section)

            if start == -1:
                continue

            section_start = start + len(section)

            remaining = answer[section_start:]

            next_section = SECTION_PATTERN.search(
                remaining
            )

            if next_section:
                content = remaining[
                    :next_section.start()
                ]
            else:
                content = remaining

            if not CITATION_PATTERN.search(content):

                uncited_sections.append(section)

                violations.append(
                    f"missing_citation:{section}"
                )

        # --------------------------------------------------
        # CLAIM-LEVEL VALIDATION
        # --------------------------------------------------

        for claim, claim_citations in self._extract_claims(
            answer
        ):

            for citation in claim_citations:

                kb_id = citation.strip("[]")

                # Already reported as unsupported citation.
                if kb_id not in retrieved_by_id:
                    continue

                article = retrieved_by_id[kb_id]

                if not self._claim_supported(
                    claim,
                    article,
                ):

                    unsupported_claims.append(
                        f"{kb_id}:{claim}"
                    )

                    violations.append(
                        f"unsupported_claim:{kb_id}:{claim}"
                    )

        # --------------------------------------------------
        # UNCITED FACTUAL CLAIMS
        # --------------------------------------------------

        for claim in self._extract_factual_claims_without_citation(
            answer
        ):

            violations.append(
                f"uncited_claim:{claim}"
            )

        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        return CitationValidationResult(
            passed=len(violations) == 0,
            citations=citations,
            unsupported_citations=unsupported_citations,
            uncited_sections=uncited_sections,
            unsupported_claims=unsupported_claims,
            violations=violations,
        )

    # ======================================================
    # CLAIM EXTRACTION
    # ======================================================

    def _extract_claims(
        self,
        answer: str,
    ):

        claims = []

        sections = self._split_sections(answer)

        for content in sections.values():

            for sentence in self._split_sentences(
                content
            ):

                citations = CITATION_PATTERN.findall(
                    sentence
                )

                if not citations:
                    continue

                claim = CITATION_PATTERN.sub(
                    "",
                    sentence,
                ).strip()

                claim = self._clean_claim(claim)

                if not claim:
                    continue

                claims.append(
                    (
                        claim,
                        sorted(set(citations)),
                    )
                )

        return claims

    def _extract_factual_claims_without_citation(
        self,
        answer: str,
    ):

        claims = []

        sections = self._split_sections(answer)

        for section_name, content in sections.items():

            # Agent response is intentionally checked too.
            # Any factual statement there should have evidence.
            for sentence in self._split_sentences(
                content
            ):

                if CITATION_PATTERN.search(sentence):
                    continue

                sentence = self._clean_claim(sentence)

                if not sentence:
                    continue

                if not self._looks_like_factual_claim(
                    sentence
                ):
                    continue

                claims.append(sentence)

        return claims

    def _split_sections(
        self,
        answer: str,
    ) -> Dict[str, str]:

        sections = {}

        matches = list(
            re.finditer(
                r"(Likely issue|Recommended troubleshooting steps|"
                r"Escalation condition|Agent response):",
                answer,
                re.IGNORECASE,
            )
        )

        for index, match in enumerate(matches):

            name = match.group(1).strip()

            start = match.end()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(answer)

            sections[name] = answer[start:end].strip()

        return sections

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:

        # Treat numbered list items as individual claims.
        text = re.sub(
            r"\n\s*\d+\.\s*",
            ". ",
            text,
        )

        # Treat bullet points as individual claims.
        text = re.sub(
            r"\n\s*[-*]\s*",
            ". ",
            text,
        )

        parts = re.split(
            r"(?<=[.!?])\s+|(?<=;)\s+",
            text,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    # ======================================================
    # CLAIM SUPPORT
    # ======================================================

    def _claim_supported(
        self,
        claim: str,
        article: dict,
    ) -> bool:

        evidence_text = self._article_to_text(
            article
        )

        claim_normalized = self._normalize_text(
            claim
        )

        evidence_normalized = self._normalize_text(
            evidence_text
        )

        if not claim_normalized:
            return False

        if not evidence_normalized:
            return False

        # --------------------------------------------------
        # Exact normalized phrase
        # --------------------------------------------------

        if claim_normalized in evidence_normalized:
            return True

        # --------------------------------------------------
        # Token comparison
        # --------------------------------------------------

        claim_tokens = self._meaningful_tokens(
            claim_normalized
        )

        evidence_tokens = self._meaningful_tokens(
            evidence_normalized
        )

        if not claim_tokens:
            return False

        if not evidence_tokens:
            return False

        overlap = claim_tokens & evidence_tokens

        overlap_ratio = (
            len(overlap)
            / len(claim_tokens)
        )

        # --------------------------------------------------
        # Important technical values
        # --------------------------------------------------

        claim_numbers = self._extract_technical_values(
            claim
        )

        evidence_numbers = self._extract_technical_values(
            evidence_text
        )

        if claim_numbers:

            if not claim_numbers.issubset(
                evidence_numbers
            ):
                return False

        # --------------------------------------------------
        # Strong overlap
        # --------------------------------------------------

        if overlap_ratio >= 0.60:
            return True

        # --------------------------------------------------
        # Short claims need stricter matching.
        # --------------------------------------------------

        if len(claim_tokens) <= 5:

            return len(overlap) >= max(
                2,
                len(claim_tokens) - 1,
            )

        # --------------------------------------------------
        # Medium/long claims
        # --------------------------------------------------

        if len(claim_tokens) <= 10:

            return (
                len(overlap) >= 5
                and overlap_ratio >= 0.50
            )

        # --------------------------------------------------
        # Long claims
        # --------------------------------------------------

        return (
            len(overlap) >= 7
            and overlap_ratio >= 0.50
        )

    # ======================================================
    # ARTICLE SERIALIZATION
    # ======================================================

    def _article_to_text(
        self,
        article: dict,
    ) -> str:

        parts = []

        def collect(value):

            if value is None:
                return

            if isinstance(value, str):

                parts.append(value)

            elif isinstance(value, dict):

                for key, item in value.items():

                    parts.append(str(key))

                    collect(item)

            elif isinstance(value, list):

                for item in value:
                    collect(item)

            else:

                parts.append(str(value))

        collect(article)

        return " ".join(parts)

    # ======================================================
    # NORMALIZATION
    # ======================================================

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        text = text.lower()

        # Normalize Unicode punctuation/dashes.
        replacements = {
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
            "\u00a0": " ",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Normalize common contractions.
        text = text.replace(
            "customer's",
            "customer",
        )

        text = text.replace(
            "node's",
            "node",
        )

        # Normalize numbers/units.
        text = re.sub(
            r"(\d+)\s*%",
            r"\1 percent",
            text,
        )

        text = re.sub(
            r"(\d+)\s*dBm",
            r"\1 dbm",
            text,
            flags=re.IGNORECASE,
        )

        # Keep letters, numbers and hyphens.
        text = re.sub(
            r"[^a-z0-9%.\- ]+",
            " ",
            text,
        )

        # Collapse whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def _meaningful_tokens(
        self,
        text: str,
    ) -> Set[str]:

        normalized = self._normalize_text(
            text
        )

        tokens = re.findall(
            r"\b[a-z0-9][a-z0-9._%-]*\b",
            normalized,
        )

        result = set()

        for token in tokens:

            if token in self.STOPWORDS:
                continue

            if len(token) <= 2 and not token.isdigit():
                continue

            result.add(token)

        return result

    # ======================================================
    # TECHNICAL VALUES
    # ======================================================

    def _extract_technical_values(
        self,
        text: str,
    ) -> Set[str]:

        normalized = self._normalize_text(
            text
        )

        values = set()

        # Percentages
        values.update(
            re.findall(
                r"\b\d+(?:\.\d+)?\s*(?:percent|%)",
                normalized,
            )
        )

        # dBm values
        values.update(
            re.findall(
                r"-?\s*\d+(?:\.\d+)?\s*dbm",
                normalized,
            )
        )

        # Time ranges such as 18:00-22:00
        values.update(
            re.findall(
                r"\b\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\b",
                normalized,
            )
        )

        # Durations such as 48 hours / 7 days
        values.update(
            re.findall(
                r"\b\d+\s+(?:hours?|days?|minutes?|weeks?)\b",
                normalized,
            )
        )

        return {
            re.sub(
                r"\s+",
                "",
                value,
            )
            for value in values
        }

    # ======================================================
    # CLAIM CLEANING
    # ======================================================

    def _clean_claim(
        self,
        claim: str,
    ) -> str:

        claim = claim.strip()

        claim = re.sub(
            r"^[\s:;,\-–—]+",
            "",
            claim,
        )

        claim = re.sub(
            r"[\s:;,\-–—]+$",
            "",
            claim,
        )

        return claim.strip()

    def _looks_like_factual_claim(
        self,
        sentence: str,
    ) -> bool:

        normalized = sentence.lower().strip()

        if not normalized:
            return False

        # Questions are not necessarily factual claims.
        if normalized.endswith("?"):
            return False

        # Headings / fragments.
        if len(normalized.split()) < 5:
            return False

        # Clearly conversational requests.
        if normalized.startswith(
            (
                "thank you",
                "please let us know",
                "please provide",
                "please confirm",
            )
        ):
            return False

        return True