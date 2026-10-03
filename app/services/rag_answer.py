import logging

from app.services.citation_validator import CitationValidator
from app.core.guardrails import OutputGuardrail
from app.services.rag import RAGContextBuilder


logger = logging.getLogger("app.rag")


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

    def answer(self, complaint: str) -> str:

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

- Every factual claim derived from the knowledge base MUST include
  an inline citation in the exact format [KB-XXXX].
- Citations MUST refer only to KB articles included in the retrieved
  knowledge-base context.
- Do not cite a KB article that was not retrieved.
- Do not invent KB IDs.
- Do not invent troubleshooting steps, causes, policies, thresholds,
  products, or technical values.
- Do not claim that any diagnostic check, test, escalation, ticket,
  or action has already been performed.
- Do not promise that an action will be performed.
- Clearly distinguish likely causes from confirmed facts.
- If the retrieved evidence is insufficient, explicitly say that the
  available knowledge-base evidence is insufficient.
- Preserve technical values and thresholds exactly as provided.
- Keep the answer concise and suitable for a support agent.

CUSTOMER COMPLAINT:
{complaint}

RETRIEVED KNOWLEDGE-BASE CONTEXT:
{context}

REQUIRED OUTPUT FORMAT:

Likely issue:
State the likely issue using only retrieved evidence.
Include at least one citation such as [KB-1001].

Recommended troubleshooting steps:
Give only troubleshooting steps explicitly supported by the retrieved
KB articles.
Each factual recommendation must include the relevant KB citation.

Escalation condition:
Give only escalation conditions supported by the retrieved KB.
Include the relevant KB citation.

Agent response:
Write a short customer-facing response grounded in the retrieved KB.
Include citations for factual claims.
Do not say that anything has already been checked, fixed, investigated,
or performed.
Do not promise that an action will be performed.
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

        return answer