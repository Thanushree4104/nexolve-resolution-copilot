import logging

from app.core.guardrails import OutputGuardrail
from app.core.logging import trace_id_ctx
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

    def answer(self, complaint: str) -> str:
        retrieved = self.retriever.search(
            complaint,
            top_k=self.context_builder.max_articles,
        )

        context = self.context_builder.build_context(
            complaint,
            retrieved,
        )

        logger.info(
            "rag_context_built",
            extra={
                "extra_fields": {
                    "retrieved_ids": [
                        article["id"]
                        for article in retrieved
                    ],
                    "article_count": len(retrieved),
                }
            },
        )

        prompt = f"""
You are a telecom support resolution assistant.

Use the knowledge-base context below to help resolve the customer's complaint.

IMPORTANT RULES:
- Use only information explicitly supported by the knowledge-base context.
- Do not invent troubleshooting steps, policies, causes, thresholds, products, or technical values.
- Do not claim that any diagnostic check, test, escalation, ticket, or action has been performed.
- Do not promise that an action will be performed.
- Clearly distinguish the likely cause from confirmed facts.
- If the knowledge base does not contain enough information, clearly say so.
- Give practical troubleshooting guidance in a clear order.
- Mention escalation conditions when relevant.
- Keep the response concise and suitable for a support agent.
- Preserve technical values and thresholds exactly as provided in the knowledge base.

CUSTOMER COMPLAINT:
{complaint}

KNOWLEDGE BASE CONTEXT:
{context}

Provide exactly these sections:

Likely issue:
State the likely issue and cite the relevant KB article ID.

Recommended troubleshooting steps:
Give only steps supported by the KB.

Escalation condition:
Give the relevant escalation conditions from the KB.

Agent response:
Write a short customer-facing response.
Do NOT say that the agent has already checked anything.
Do NOT promise that the agent will perform anything.
Use wording such as "The recommended next step is..." or "This pattern is consistent with..."
"""

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

        guardrail_result = self.guardrail.validate(
            answer=answer,
            retrieved_articles=retrieved,
        )

        if not guardrail_result.passed:
            logger.warning(
                "rag_guardrail_failed",
                extra={
                    "extra_fields": {
                        "violations": guardrail_result.violations,
                        "retrieved_ids": [
                            article["id"]
                            for article in retrieved
                        ],
                    }
                },
            )

            raise ValueError(
                "Generated answer failed output guardrails: "
                + ", ".join(
                    guardrail_result.violations
                )
            )

        logger.info(
            "rag_guardrail_passed",
            extra={
                "extra_fields": {
                    "retrieved_ids": [
                        article["id"]
                        for article in retrieved
                    ],
                    "violation_count": 0,
                }
            },
        )

        return answer