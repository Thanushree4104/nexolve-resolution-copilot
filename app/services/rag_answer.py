import logging
import time
from dataclasses import dataclass
from typing import Optional

from app.services.citation_validator import CitationValidator
from app.core.guardrails import OutputGuardrail
from app.services.rag import RAGContextBuilder


logger = logging.getLogger("app.rag")


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
        self.citation_validator = CitationValidator()

    def _generate(self, complaint: str) -> RAGAnswerResult:

        total_start = time.perf_counter()

        # --------------------------------------------------
        # RETRIEVAL
        # --------------------------------------------------

        retrieved = self.retriever.search(
            complaint,
            top_k=self.context_builder.max_articles,
        )

        retrieved_ids = [
            article["id"]
            for article in retrieved
        ]

        # --------------------------------------------------
        # BUILD RAG CONTEXT
        # --------------------------------------------------

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

        # --------------------------------------------------
        # PROMPT
        # --------------------------------------------------

        prompt = f"""
You are a telecom support resolution assistant.

Use ONLY the knowledge-base context provided below.

IMPORTANT GROUNDING RULES:

- Use only information explicitly supported by the retrieved
  knowledge-base context.
- Every factual claim derived from the knowledge base MUST include
  an inline citation in the exact format [KB-XXXX].
- Citations MUST refer only to KB articles included in the retrieved
  knowledge-base context.
- Never cite a KB article that was not retrieved.
- Never invent KB IDs.
- Never invent troubleshooting steps, causes, policies, thresholds,
  products, technical values, or escalation conditions.
- Preserve technical values and thresholds exactly as provided.
- Clearly distinguish likely causes from confirmed facts.
- If the evidence is insufficient, explicitly state that the available
  knowledge-base evidence is insufficient.
- Do not claim that any diagnostic check, test, investigation,
  escalation, ticket, repair, or action has already been performed.
- Do not promise that an action will be performed.
- Do not use future-action commitments such as:
  "I will", "we will", "I'll", "we'll".
- Do not say that the support team, agent, engineering team, or any
  other party will perform an action.
- Describe actions only as recommendations or conditions.
- Keep the answer concise and suitable for a support agent.

CUSTOMER COMPLAINT:
{complaint}

RETRIEVED KNOWLEDGE-BASE CONTEXT:
{context}

REQUIRED OUTPUT FORMAT:

Likely issue:
State the likely issue using only retrieved evidence.
Every factual statement MUST include the relevant KB citation.

Recommended troubleshooting steps:
Give only troubleshooting steps explicitly supported by the
retrieved KB articles.

Every factual troubleshooting recommendation MUST include
the relevant KB citation.

Escalation condition:
Give only escalation conditions explicitly supported by the
retrieved KB articles.

Every factual escalation condition MUST include the relevant
KB citation.

Agent response:
Write a short customer-facing response grounded ONLY in the
retrieved KB evidence.

STRICT AGENT RESPONSE RULES:

- Every factual statement MUST include a KB citation.
- Do not make any uncited factual statement.
- Do not introduce information that does not appear in the retrieved KB.
- Do not say that anything has already been checked.
- Do not say that anything has already been fixed.
- Do not say that anything has already been investigated.
- Do not say that a test has already been performed.
- Do not say that a ticket has already been created.
- Do not promise that an action will happen.
- Do not use:
  "I will"
  "we will"
  "I'll"
  "we'll"
  "I am checking"
  "we are checking"
  "I'll keep you updated"
  or equivalent future-action commitments.

Use neutral recommendation wording such as:

"The recommended next step is..."
"The available KB guidance recommends..."
"This pattern is consistent with..."
"Escalation is appropriate if..."

If the evidence is insufficient, say:

"The available knowledge-base evidence is insufficient to determine
the cause or recommended next step."

Remember:
A factual sentence without a citation is invalid.
A citation to a KB article that was not retrieved is invalid.
An invented KB ID is invalid.
An unsupported factual claim is invalid.
"""

        # --------------------------------------------------
        # LLM GENERATION
        # --------------------------------------------------

        response = self.provider.complete(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_tokens=1000,
        )

        answer = response.text

        # --------------------------------------------------
        # OUTPUT GUARDRAIL
        # --------------------------------------------------

        guardrail_result = self.guardrail.validate(
            answer=answer,
            retrieved_articles=retrieved,
        )

        if not guardrail_result.passed:

            logger.warning(
                "rag_guardrail_failed",
                extra={
                    "extra_fields": {
                        "violations": (
                            guardrail_result.violations
                        ),
                        "retrieved_ids": retrieved_ids,
                    }
                },
            )

            raise ValueError(
                "Generated answer failed output guardrails: "
                + ", ".join(
                    guardrail_result.violations
                )
            )

        # --------------------------------------------------
        # CITATION VALIDATION
        # --------------------------------------------------

        citation_result = self.citation_validator.validate(
            answer=answer,
            retrieved_articles=retrieved,
        )

        if not citation_result.passed:

            logger.warning(
                "rag_citation_validation_failed",
                extra={
                    "extra_fields": {
                        "citations": (
                            citation_result.citations
                        ),
                        "unsupported_citations": (
                            citation_result.unsupported_citations
                        ),
                        "uncited_sections": (
                            citation_result.uncited_sections
                        ),
                        "unsupported_claims": (
                            getattr(
                                citation_result,
                                "unsupported_claims",
                                [],
                            )
                        ),
                        "violations": (
                            citation_result.violations
                        ),
                        "retrieved_ids": retrieved_ids,
                    }
                },
            )

            raise ValueError(
                "Generated answer failed citation validation: "
                + ", ".join(
                    citation_result.violations
                )
            )

        # --------------------------------------------------
        # SUCCESS LOGGING
        # --------------------------------------------------

        logger.info(
            "rag_citation_validation_passed",
            extra={
                "extra_fields": {
                    "citations": (
                        citation_result.citations
                    ),
                    "citation_count": len(
                        citation_result.citations
                    ),
                    "retrieved_ids": retrieved_ids,
                }
            },
        )

        logger.info(
            "rag_guardrail_passed",
            extra={
                "extra_fields": {
                    "retrieved_ids": retrieved_ids,
                    "violation_count": 0,
                }
            },
        )

        # --------------------------------------------------
        # BUILD DETAILED RESULT
        # --------------------------------------------------

        total_latency_ms = round(
            (time.perf_counter() - total_start) * 1000,
            1,
        )

        return RAGAnswerResult(
            answer=answer,
            retrieved_articles=retrieved,
            retrieved_ids=retrieved_ids,
            citations=citation_result.citations,
            citation_result=citation_result,
            guardrail_result=guardrail_result,
            llm_model=getattr(
                response,
                "model",
                None,
            ),
            llm_cached=getattr(
                response,
                "cached",
                False,
            ),
            llm_latency_ms=getattr(
                response,
                "latency_ms",
                0.0,
            ),
            prompt_tokens=getattr(
                response,
                "prompt_tokens",
                None,
            ),
            completion_tokens=getattr(
                response,
                "completion_tokens",
                None,
            ),
            total_latency_ms=total_latency_ms,
        )

    def answer(self, complaint: str) -> str:
        """
        Backward-compatible public API.

        Existing /resolve code continues receiving
        a plain string.
        """

        result = self._generate(complaint)

        return result.answer

    def answer_with_metadata(
        self,
        complaint: str,
    ) -> RAGAnswerResult:
        """
        Evaluation and observability API.

        Returns the generated answer together with
        retrieval, citation, guardrail, and LLM metadata.
        """

        return self._generate(complaint)