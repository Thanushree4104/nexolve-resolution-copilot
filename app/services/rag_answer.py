import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from app.services.citation_validator import CitationValidator
from app.core.guardrails import OutputGuardrail
from app.services.rag import RAGContextBuilder


logger = logging.getLogger("app.rag")


# ============================================================
# REQUIRED OUTPUT FORMAT
# ============================================================

REQUIRED_SECTIONS = [
    "Likely issue:",
    "Recommended troubleshooting steps:",
    "Escalation condition:",
    "Agent response:",
]


@dataclass
class RAGAnswerResult:
    answer: str
    retrieved_articles: list
    retrieved_ids: list
    citations: list
    citation_result: object
    guardrail_result: object
    llm_model: Optional[str]
    llm_cached: bool
    llm_latency_ms: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_latency_ms: float


class RAGAnswerService:

    def __init__(
        self,
        provider,
        retriever,
        max_articles=3,
    ):
        self.provider = provider
        self.retriever = retriever

        self.context_builder = RAGContextBuilder(
            max_articles=max_articles
        )

        self.guardrail = OutputGuardrail()

        self.citation_validator = CitationValidator(
            require_section_citations=False
        )

        self.max_articles = max_articles

    # ============================================================
    # PROMPT
    # ============================================================

    def _build_prompt(
        self,
        complaint: str,
        context: str,
    ) -> str:

        return f"""
You are a telecom support resolution assistant.

Do not mention that you are an AI.

Your task is to help resolve the customer's complaint using ONLY the
knowledge-base context provided below.

CUSTOMER COMPLAINT:
{complaint}

KNOWLEDGE-BASE CONTEXT:
{context}


============================================================
STRICT GROUNDING RULES
============================================================

1. Use ONLY information explicitly supported by the retrieved
   knowledge-base articles.

2. Do NOT invent:
   - troubleshooting steps
   - root causes
   - diagnoses
   - escalation conditions
   - technical values
   - thresholds
   - commands
   - policies
   - product behavior
   - procedures

3. Do NOT claim that any diagnostic check, test, escalation,
   ticket, configuration change, or other action has already
   been performed.

4. Do NOT make promises or commitments such as:
   - "I will..."
   - "We will..."
   - "I'll..."
   - "We'll..."
   - "I am going to..."
   - "We are going to..."

5. Do not say that you have personally checked, fixed,
   investigated, resolved, or performed an action.

6. Clearly distinguish between a likely issue and a confirmed issue.

7. Preserve technical values, thresholds, names, and procedures
   exactly as provided in the knowledge base.

8. If the knowledge base does not contain enough information to
   support a statement, do not guess.

9. Every factual statement derived from a knowledge-base article
   MUST have a citation using the format:

   [KB-XXXX]

10. The citation must be attached to the claim it supports.

11. Do not cite KB IDs that are not present in the supplied
    knowledge-base context.


============================================================
MANDATORY OUTPUT FORMAT
============================================================

Your response MUST contain EXACTLY these four section headings,
in EXACTLY this order:

Likely issue:

Recommended troubleshooting steps:

Escalation condition:

Agent response:


Do NOT rename these headings.

Do NOT remove these headings.

Do NOT add additional top-level sections.

Do NOT put a preamble before "Likely issue:".

Do NOT put a conclusion after "Agent response:".


============================================================
SECTION REQUIREMENTS
============================================================

Likely issue:
State the most likely issue supported by the knowledge base.
Do not present an unconfirmed issue as a confirmed diagnosis.
Include the appropriate KB citation.

Recommended troubleshooting steps:
Provide only troubleshooting steps explicitly supported by the
knowledge base.

Use numbered steps when appropriate.

Every factual troubleshooting step must contain its KB citation.

Escalation condition:
State the escalation condition only if the knowledge base provides
one.

Every factual escalation condition must contain its KB citation.

If the knowledge base does not specify an escalation condition,
write:

The available knowledge-base evidence is insufficient to determine
an escalation condition.

Agent response:
Write a concise, professional response that an agent could send
to the customer.

Do not claim that the agent has already performed an action.

Do not promise that the agent will perform an action.

Any factual statement derived from the knowledge base must contain
a citation.


============================================================
IMPORTANT
============================================================

If a statement cannot be directly supported by the supplied
knowledge-base context, OMIT the statement.

Do not infer missing transaction IDs, billing-system results,
diagnostic results, or investigation outcomes.

For example, this is NOT allowed unless explicitly supported:

"Unable to find matching transaction IDs in the billing system."

If the knowledge base only says that escalation is appropriate
when matching transaction IDs cannot be found, phrase it as a
conditional escalation rule:

"Escalate if matching transaction IDs cannot be found in the
billing system [KB-XXXX]."


============================================================
FINAL CHECK BEFORE RESPONDING
============================================================

Before producing the response, verify:

- All four required headings are present.
- The headings are in the correct order.
- Every KB-derived factual claim has a citation.
- Every citation refers to a KB article in the supplied context.
- No unsupported information has been added.
- No action is claimed as already performed.
- No future action or commitment is promised.
- There is no preamble or conclusion outside the four sections.

Return ONLY the four required sections.
"""

    # ============================================================
    # FORMAT CHECK
    # ============================================================

    def _has_required_sections(
        self,
        answer: str,
    ) -> bool:

        if not answer:
            return False

        positions = []

        for section in REQUIRED_SECTIONS:

            match = re.search(
                rf"(?im)^\s*{re.escape(section)}\s*$",
                answer,
            )

            if not match:
                return False

            positions.append(
                match.start()
            )

        return positions == sorted(positions)

    # ============================================================
    # FORMAT REPAIR PROMPT
    # ============================================================

    def _build_format_repair_prompt(
        self,
        answer: str,
        context: str,
    ) -> str:

        return f"""
You are formatting an existing telecom support answer.

Your task is ONLY to repair the structure and remove unsupported
content.

Do NOT add new information.

Do NOT invent troubleshooting steps.

Do NOT invent causes.

Do NOT invent escalation conditions.

Do NOT remove valid KB citations.

Do NOT change KB IDs.

Do NOT introduce any KB ID that does not already appear in the
answer or supplied knowledge-base context.

Do NOT introduce future promises or action commitments.

Remove unsupported factual claims.

Avoid phrases such as:

"I will..."
"We will..."
"I'll..."
"We'll..."
"I am going to..."
"We are going to..."
"I have checked..."
"I have fixed..."
"I have resolved..."
"I am investigating..."

Reformat the answer so that it contains EXACTLY these four sections,
in this exact order:

Likely issue:

Recommended troubleshooting steps:

Escalation condition:

Agent response:

Preserve the original factual meaning.

If a factual statement already has a KB citation, keep that citation
attached to the statement.

If information is insufficient for a section, use:

The available knowledge-base evidence is insufficient to determine
this.

For escalation specifically, use:

The available knowledge-base evidence is insufficient to determine
an escalation condition.

Do not add a preamble.

Do not add a conclusion.

Return ONLY the four required sections.

KNOWLEDGE-BASE CONTEXT:
{context}

ORIGINAL ANSWER:
{answer}
"""

    # ============================================================
    # CITATION / GROUNDING REPAIR PROMPT
    # ============================================================

    def _build_citation_repair_prompt(
        self,
        answer: str,
        context: str,
        errors: list,
    ) -> str:

        error_text = ", ".join(
            str(error)
            for error in (errors or [])
        )

        return f"""
You are repairing a telecom support answer that failed citation
or grounding validation.

Your ONLY task is to rewrite the answer so that every factual
claim is directly supported by the supplied knowledge-base context.

VALIDATION ERRORS:
{error_text}

KNOWLEDGE-BASE CONTEXT:
{context}

ORIGINAL ANSWER:
{answer}


============================================================
STRICT REPAIR RULES
============================================================

1. Do NOT add new information.

2. Do NOT invent causes.

3. Do NOT invent troubleshooting steps.

4. Do NOT invent escalation conditions.

5. Do NOT invent diagnostic results.

6. Do NOT invent billing-system results.

7. Do NOT claim that an action has already happened.

8. Do NOT make future commitments or promises.

9. Preserve valid KB citations.

10. Use ONLY KB IDs that exist in the supplied context.

11. Every factual KB-derived statement MUST contain its
    appropriate citation.

12. If the original answer contains a factual statement that
    cannot be supported by the supplied context, REMOVE that
    statement.

13. Conditional escalation rules are allowed ONLY when explicitly
    supported by the knowledge base.

For example:

Allowed:
"Escalate if matching transaction IDs cannot be found in the
billing system [KB-1018]."

Not allowed:
"Matching transaction IDs cannot be found in the billing system."

14. Do not turn a conditional KB statement into a confirmed event.

15. Do not claim that the customer has already experienced a
    specific technical condition unless the knowledge base and
    complaint explicitly support that conclusion.

16. The answer MUST contain exactly these four sections:

Likely issue:

Recommended troubleshooting steps:

Escalation condition:

Agent response:

17. Do not add a preamble or conclusion.

Return ONLY the repaired four-section answer.
"""

    # ============================================================
    # GUARDRAIL REPAIR PROMPT
    # ============================================================

    def _build_guardrail_repair_prompt(
        self,
        answer: str,
        context: str,
        violations: list,
    ) -> str:

        violation_text = ", ".join(
            str(value)
            for value in (violations or [])
        )

        return f"""
You are repairing a telecom support answer that failed an output
safety or grounding guardrail.

Your ONLY task is to rewrite the answer so that it passes the
guardrail while preserving supported factual information.

GUARDRAIL VIOLATIONS:
{violation_text}

KNOWLEDGE-BASE CONTEXT:
{context}

ORIGINAL ANSWER:
{answer}


============================================================
REPAIR RULES
============================================================

1. Do NOT add new information.

2. Do NOT invent causes, procedures, troubleshooting steps,
   escalation conditions, or technical details.

3. Preserve every valid KB citation.

4. Do NOT introduce any KB ID that is not supported by the
   knowledge-base context.

5. Do NOT claim that an action has already happened.

6. Do NOT make future commitments or promises.

7. Replace forbidden action/commitment language with neutral
   instructions or conditional language.

For example:

BAD:
"I will investigate the issue."

GOOD:
"Please provide the relevant details so the issue can be
investigated."

BAD:
"We will arrange a technician visit."

GOOD:
"A technician visit may be appropriate if the escalation
condition is met."

BAD:
"I'll check the transaction details."

GOOD:
"Check the transaction details."

8. Keep the distinction between a likely issue and a confirmed issue.

9. Every KB-derived factual statement must retain its [KB-XXXX]
   citation.

10. Remove unsupported factual claims rather than guessing.

11. The answer MUST contain exactly these four sections:

Likely issue:

Recommended troubleshooting steps:

Escalation condition:

Agent response:

12. Do not add a preamble or conclusion.

Return ONLY the repaired four-section answer.
"""

    # ============================================================
    # GENERATION HELPER
    # ============================================================

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 1200,
    ):
        return self.provider.complete(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )

    # ============================================================
    # RESPONSE TEXT EXTRACTION
    # ============================================================

    def _extract_response_text(
        self,
        response,
    ) -> str:

        if response is None:
            return ""

        raw_text = getattr(
            response,
            "text",
            response,
        )

        if raw_text is None:
            return ""

        return str(
            raw_text
        ).strip()

    # ============================================================
    # CITATION VALIDATION
    # ============================================================

    def _validate_citations(
        self,
        answer: str,
        retrieved: list,
    ):

        result = self.citation_validator.validate(
            answer,
            retrieved_articles=retrieved,
        )

        if hasattr(result, "passed"):
            passed = result.passed

        elif hasattr(result, "valid"):
            passed = result.valid

        elif hasattr(result, "is_valid"):
            passed = result.is_valid

        else:
            passed = bool(result)

        if not passed:

            violations = getattr(
                result,
                "violations",
                None,
            )

            if not violations:
                violations = getattr(
                    result,
                    "errors",
                    None,
                )

            if violations:

                detail = ", ".join(
                    str(value)
                    for value in violations
                )

            else:

                detail = str(
                    result
                )

            raise RuntimeError(
                "Generated answer failed citation validation: "
                + detail
            )

        return result

    # ============================================================
    # GUARDRAIL VALIDATION
    # ============================================================

    def _validate_guardrail(
        self,
        answer: str,
        retrieved: list,
    ):

        return self.guardrail.validate(
            answer,
            retrieved_articles=retrieved,
        )

    # ============================================================
    # SAFE FALLBACK
    # ============================================================

    def _insufficient_evidence_answer(
        self,
    ) -> str:

        return (
            "Likely issue:\n"
            "The available knowledge-base evidence is insufficient "
            "to determine the likely issue.\n\n"

            "Recommended troubleshooting steps:\n"
            "The available knowledge-base evidence is insufficient "
            "to determine appropriate troubleshooting steps.\n\n"

            "Escalation condition:\n"
            "The available knowledge-base evidence is insufficient "
            "to determine an escalation condition.\n\n"

            "Agent response:\n"
            "The available knowledge-base evidence is insufficient "
            "to determine the appropriate resolution from the "
            "available knowledge base."
        )

    # ============================================================
    # DETERMINISTIC GUARDRAIL FALLBACK
    # ============================================================

    def _guardrail_fallback_answer(
        self,
        retrieved: list,
    ) -> str:

        """
        Deterministic final fallback.

        This intentionally makes NO KB-derived factual claims,
        therefore it does not need KB citations.

        It is used only when the generated answer cannot be made
        safe and grounded after repair attempts.
        """

        return (
            "Likely issue:\n"
            "The available knowledge-base evidence is insufficient "
            "to safely determine the likely issue.\n\n"

            "Recommended troubleshooting steps:\n"
            "The available knowledge-base evidence is insufficient "
            "to safely determine appropriate troubleshooting steps.\n\n"

            "Escalation condition:\n"
            "The available knowledge-base evidence is insufficient "
            "to determine an escalation condition.\n\n"

            "Agent response:\n"
            "The available knowledge-base evidence does not support "
            "a safe, specific resolution for this complaint."
        )

    # ============================================================
    # GET VALIDATION VIOLATIONS
    # ============================================================

    def _get_validation_violations(
        self,
        result,
    ) -> list:

        if result is None:
            return []

        violations = getattr(
            result,
            "violations",
            None,
        )

        if violations:
            return list(
                violations
            )

        errors = getattr(
            result,
            "errors",
            None,
        )

        if errors:
            return list(
                errors
            )

        return []

    # ============================================================
    # GET PASS STATUS
    # ============================================================

    def _get_passed(
        self,
        result,
    ) -> bool:

        if result is None:
            return False

        if hasattr(
            result,
            "passed",
        ):
            return bool(
                result.passed
            )

        if hasattr(
            result,
            "valid",
        ):
            return bool(
                result.valid
            )

        if hasattr(
            result,
            "is_valid",
        ):
            return bool(
                result.is_valid
            )

        return bool(result)

    # ============================================================
    # MAIN GENERATION
    # ============================================================

    def _generate(
        self,
        complaint: str,
    ) -> RAGAnswerResult:

        total_start = time.perf_counter()

        # ========================================================
        # RETRIEVAL
        # ========================================================

        retrieved = self.retriever.search(
            complaint,
            top_k=self.context_builder.max_articles,
        )

        retrieved = retrieved or []

        retrieved_ids = [
            article.get("id", "")
            for article in retrieved
            if isinstance(article, dict)
        ]

        logger.info(
            "rag_retrieval_completed",
            extra={
                "extra_fields": {
                    "retrieved_ids": retrieved_ids,
                    "article_count": len(retrieved),
                }
            },
        )

        # ========================================================
        # NOTE:
        #
        # Feedback reranking is already performed inside
        # HybridRetriever.search().
        #
        # Do NOT rerank here again.
        # ========================================================

        # ========================================================
        # NO RETRIEVAL
        # ========================================================

        if not retrieved:

            answer = self._insufficient_evidence_answer()

            total_latency_ms = round(
                (
                    time.perf_counter()
                    - total_start
                )
                * 1000,
                1,
            )

            return RAGAnswerResult(
                answer=answer,
                retrieved_articles=[],
                retrieved_ids=[],
                citations=[],
                citation_result=None,
                guardrail_result=None,
                llm_model=None,
                llm_cached=False,
                llm_latency_ms=0.0,
                prompt_tokens=None,
                completion_tokens=None,
                total_latency_ms=total_latency_ms,
            )

        # ========================================================
        # BUILD RAG CONTEXT
        # ========================================================

        context = self.context_builder.build_context(
            complaint,
            retrieved,
        )

        logger.info(
            "rag_context_built",
            extra={
                "extra_fields": {
                    "retrieved_ids": retrieved_ids,
                    "article_count": len(retrieved),
                }
            },
        )

        # ========================================================
        # BUILD PROMPT
        # ========================================================

        prompt = self._build_prompt(
            complaint,
            context,
        )

        # ========================================================
        # LLM GENERATION
        # ========================================================

        llm_start = time.perf_counter()

        try:

            response = self._call_llm(
                prompt,
                max_tokens=1200,
            )

        except Exception as exc:

            raise RuntimeError(
                "RAG answer generation failed: "
                + str(exc)[:300]
            ) from exc

        llm_latency_ms = round(
            (
                time.perf_counter()
                - llm_start
            )
            * 1000,
            1,
        )

        answer = self._extract_response_text(
            response
        )

        if not answer:

            raise RuntimeError(
                "Empty response from LLM"
            )

        # ========================================================
        # FORMAT REPAIR
        # ========================================================

        if not self._has_required_sections(
            answer
        ):

            logger.warning(
                "rag_output_missing_required_sections",
                extra={
                    "extra_fields": {
                        "retrieved_ids": retrieved_ids,
                    }
                },
            )

            repair_prompt = (
                self._build_format_repair_prompt(
                    answer,
                    context,
                )
            )

            repair_start = time.perf_counter()

            try:

                repaired_response = self._call_llm(
                    repair_prompt,
                    max_tokens=1200,
                )

                repaired_answer = (
                    self._extract_response_text(
                        repaired_response
                    )
                )

                if repaired_answer:
                    answer = repaired_answer

            except Exception as exc:

                logger.warning(
                    "rag_format_repair_failed",
                    extra={
                        "extra_fields": {
                            "error": str(exc),
                        }
                    },
                )

            repair_latency_ms = round(
                (
                    time.perf_counter()
                    - repair_start
                )
                * 1000,
                1,
            )

            llm_latency_ms = round(
                llm_latency_ms
                + repair_latency_ms,
                1,
            )

        # ========================================================
        # FINAL STRUCTURE CHECK
        # ========================================================

        if not self._has_required_sections(
            answer
        ):

            logger.error(
                "rag_output_format_repair_exhausted",
                extra={
                    "extra_fields": {
                        "retrieved_ids": retrieved_ids,
                    }
                },
            )

            answer = self._guardrail_fallback_answer(
                retrieved
            )

        # ========================================================
        # CITATION VALIDATION
        # ========================================================

        logger.info(
            "rag_answer_before_citation_validation",
            extra={
                "extra_fields": {
                    "retrieved_ids": retrieved_ids,
                }
            },
        )

        citation_result = None
        citation_passed = False

        try:

            citation_result = self._validate_citations(
                answer,
                retrieved,
            )

            citation_passed = True

        except RuntimeError as exc:

            citation_errors = [str(exc)]

            logger.warning(
                "rag_citation_validation_failed_attempting_repair",
                extra={
                    "extra_fields": {
                        "error": str(exc),
                        "retrieved_ids": retrieved_ids,
                    }
                },
            )

            # ====================================================
            # IMPORTANT:
            #
            # Do NOT immediately fallback.
            #
            # First give the LLM a chance to remove/repair the
            # unsupported claim while preserving valid content.
            # ====================================================

            citation_repair_prompt = (
                self._build_citation_repair_prompt(
                    answer,
                    context,
                    citation_errors,
                )
            )

            repair_start = time.perf_counter()

            try:

                repaired_response = self._call_llm(
                    citation_repair_prompt,
                    max_tokens=1200,
                )

                repaired_answer = (
                    self._extract_response_text(
                        repaired_response
                    )
                )

                if repaired_answer:

                    answer = repaired_answer

            except Exception as repair_exc:

                logger.warning(
                    "rag_citation_repair_failed",
                    extra={
                        "extra_fields": {
                            "error": str(repair_exc),
                        }
                    },
                )

            repair_latency_ms = round(
                (
                    time.perf_counter()
                    - repair_start
                )
                * 1000,
                1,
            )

            llm_latency_ms = round(
                llm_latency_ms
                + repair_latency_ms,
                1,
            )

            # ====================================================
            # RE-CHECK STRUCTURE
            # ====================================================

            if not self._has_required_sections(
                answer
            ):

                logger.warning(
                    "rag_citation_repair_missing_sections",
                    extra={
                        "extra_fields": {
                            "retrieved_ids": retrieved_ids,
                        }
                    },
                )

                answer = self._guardrail_fallback_answer(
                    retrieved
                )

            # ====================================================
            # RE-CHECK CITATIONS
            # ====================================================

            try:

                citation_result = (
                    self._validate_citations(
                        answer,
                        retrieved,
                    )
                )

                citation_passed = True

                logger.info(
                    "rag_citation_repair_passed",
                    extra={
                        "extra_fields": {
                            "retrieved_ids": retrieved_ids,
                        }
                    },
                )

            except RuntimeError as second_exc:

                logger.error(
                    "rag_citation_repair_exhausted_using_fallback",
                    extra={
                        "extra_fields": {
                            "error": str(second_exc),
                            "retrieved_ids": retrieved_ids,
                        }
                    },
                )

                answer = self._guardrail_fallback_answer(
                    retrieved
                )

                # The deterministic fallback has no KB-derived
                # factual claims, so citation validation should
                # normally pass.

                try:

                    citation_result = (
                        self._validate_citations(
                            answer,
                            retrieved,
                        )
                    )

                    citation_passed = True

                except RuntimeError as fallback_exc:

                    logger.error(
                        "rag_citation_fallback_validation_failed",
                        extra={
                            "extra_fields": {
                                "error": str(fallback_exc),
                                "retrieved_ids": retrieved_ids,
                            }
                        },
                    )

                    citation_result = None
                    citation_passed = False

        # ========================================================
        # GUARDRAIL VALIDATION
        # ========================================================

        guardrail_result = None

        try:

            guardrail_result = self._validate_guardrail(
                answer,
                retrieved,
            )

        except Exception as exc:

            logger.error(
                "rag_guardrail_validation_exception",
                extra={
                    "extra_fields": {
                        "error": str(exc),
                        "retrieved_ids": retrieved_ids,
                    }
                },
            )

            guardrail_result = None

        # ========================================================
        # GUARDRAIL FAILURE -> REPAIR
        # ========================================================

        if guardrail_result is not None:

            guardrail_passed = self._get_passed(
                guardrail_result
            )

            if not guardrail_passed:

                violations = (
                    self._get_validation_violations(
                        guardrail_result
                    )
                )

                logger.warning(
                    "rag_guardrail_failed_attempting_repair",
                    extra={
                        "extra_fields": {
                            "violations": violations,
                            "retrieved_ids": retrieved_ids,
                        }
                    },
                )

                repair_prompt = (
                    self._build_guardrail_repair_prompt(
                        answer,
                        context,
                        violations,
                    )
                )

                repair_start = time.perf_counter()

                try:

                    repaired_response = self._call_llm(
                        repair_prompt,
                        max_tokens=1200,
                    )

                    repaired_answer = (
                        self._extract_response_text(
                            repaired_response
                        )
                    )

                    if repaired_answer:

                        answer = repaired_answer

                except Exception as exc:

                    logger.warning(
                        "rag_guardrail_repair_failed",
                        extra={
                            "extra_fields": {
                                "error": str(exc),
                            }
                        },
                    )

                repair_latency_ms = round(
                    (
                        time.perf_counter()
                        - repair_start
                    )
                    * 1000,
                    1,
                )

                llm_latency_ms = round(
                    llm_latency_ms
                    + repair_latency_ms,
                    1,
                )

                # ------------------------------------------------
                # Re-check structure
                # ------------------------------------------------

                if not self._has_required_sections(
                    answer
                ):

                    logger.warning(
                        "rag_guardrail_repair_missing_sections",
                        extra={
                            "extra_fields": {
                                "retrieved_ids": retrieved_ids,
                            }
                        },
                    )

                    answer = self._guardrail_fallback_answer(
                        retrieved
                    )

                # ------------------------------------------------
                # Re-check citations
                # ------------------------------------------------

                try:

                    citation_result = (
                        self._validate_citations(
                            answer,
                            retrieved,
                        )
                    )

                    citation_passed = True

                except RuntimeError as exc:

                    logger.warning(
                        "rag_guardrail_repair_citation_failed",
                        extra={
                            "extra_fields": {
                                "error": str(exc),
                                "retrieved_ids": retrieved_ids,
                            }
                        },
                    )

                    # One final deterministic fallback.

                    answer = self._guardrail_fallback_answer(
                        retrieved
                    )

                    try:

                        citation_result = (
                            self._validate_citations(
                                answer,
                                retrieved,
                            )
                        )

                        citation_passed = True

                    except RuntimeError:

                        citation_result = None
                        citation_passed = False

                # ------------------------------------------------
                # Final guardrail validation
                # ------------------------------------------------

                try:

                    guardrail_result = (
                        self._validate_guardrail(
                            answer,
                            retrieved,
                        )
                    )

                except Exception as exc:

                    logger.error(
                        "rag_guardrail_revalidation_failed",
                        extra={
                            "extra_fields": {
                                "error": str(exc),
                                "retrieved_ids": retrieved_ids,
                            }
                        },
                    )

                    guardrail_result = None

                final_passed = self._get_passed(
                    guardrail_result
                )

                # ------------------------------------------------
                # FINAL DETERMINISTIC FALLBACK
                # ------------------------------------------------

                if not final_passed:

                    logger.error(
                        "rag_guardrail_repair_exhausted_using_fallback",
                        extra={
                            "extra_fields": {
                                "retrieved_ids": retrieved_ids,
                            }
                        },
                    )

                    answer = self._guardrail_fallback_answer(
                        retrieved
                    )

                    # Validate deterministic fallback.

                    try:

                        guardrail_result = (
                            self._validate_guardrail(
                                answer,
                                retrieved,
                            )
                        )

                    except Exception as exc:

                        logger.error(
                            "rag_guardrail_fallback_validation_failed",
                            extra={
                                "extra_fields": {
                                    "error": str(exc),
                                    "retrieved_ids": retrieved_ids,
                                }
                            },
                        )

                        guardrail_result = None

        # ========================================================
        # FINAL SAFETY CHECK
        # ========================================================

        if guardrail_result is not None:

            final_passed = self._get_passed(
                guardrail_result
            )

            if not final_passed:

                logger.error(
                    "rag_guardrail_final_failure",
                    extra={
                        "extra_fields": {
                            "retrieved_ids": retrieved_ids,
                            "violations": (
                                self._get_validation_violations(
                                    guardrail_result
                                )
                            ),
                        }
                    },
                )

                answer = self._insufficient_evidence_answer()

                try:

                    guardrail_result = (
                        self._validate_guardrail(
                            answer,
                            retrieved,
                        )
                    )

                except Exception:

                    guardrail_result = None

        else:

            logger.warning(
                "rag_guardrail_result_unavailable",
                extra={
                    "extra_fields": {
                        "retrieved_ids": retrieved_ids,
                    }
                },
            )

        # ========================================================
        # GUARDRAIL SUCCESS LOG
        # ========================================================

        if guardrail_result is not None:

            passed = self._get_passed(
                guardrail_result
            )

            if passed:

                logger.info(
                    "rag_guardrail_validation_passed",
                    extra={
                        "extra_fields": {
                            "retrieved_ids": retrieved_ids,
                        }
                    },
                )

        # ========================================================
        # EXTRACT CITATIONS
        # ========================================================

        citations = []

        if citation_result is not None:

            citations = getattr(
                citation_result,
                "citations",
                [],
            )

            citations = list(
                citations or []
            )

        # ========================================================
        # MODEL METADATA
        # ========================================================

        llm_model = getattr(
            response,
            "model",
            None,
        )

        llm_cached = bool(
            getattr(
                response,
                "cached",
                False,
            )
        )

        prompt_tokens = getattr(
            response,
            "prompt_tokens",
            None,
        )

        completion_tokens = getattr(
            response,
            "completion_tokens",
            None,
        )

        # Some providers expose usage as a nested object.

        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage is not None:

            if prompt_tokens is None:

                prompt_tokens = getattr(
                    usage,
                    "prompt_tokens",
                    None,
                )

            if completion_tokens is None:

                completion_tokens = getattr(
                    usage,
                    "completion_tokens",
                    None,
                )

        # ========================================================
        # FINAL RESULT
        # ========================================================

        total_latency_ms = round(
            (
                time.perf_counter()
                - total_start
            )
            * 1000,
            1,
        )

        return RAGAnswerResult(
            answer=answer,
            retrieved_articles=retrieved,
            retrieved_ids=retrieved_ids,
            citations=citations,
            citation_result=citation_result,
            guardrail_result=guardrail_result,
            llm_model=llm_model,
            llm_cached=llm_cached,
            llm_latency_ms=llm_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_latency_ms=total_latency_ms,
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def answer(
        self,
        complaint: str,
    ) -> RAGAnswerResult:

        complaint = str(
            complaint or ""
        ).strip()

        if not complaint:

            answer = self._insufficient_evidence_answer()

            return RAGAnswerResult(
                answer=answer,
                retrieved_articles=[],
                retrieved_ids=[],
                citations=[],
                citation_result=None,
                guardrail_result=None,
                llm_model=None,
                llm_cached=False,
                llm_latency_ms=0.0,
                prompt_tokens=None,
                completion_tokens=None,
                total_latency_ms=0.0,
            )

        return self._generate(
            complaint
        )