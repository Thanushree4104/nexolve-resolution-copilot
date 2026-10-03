from app.services.rag import RAGContextBuilder


class RAGAnswerService:
    def __init__(self, provider, retriever, max_articles=3):
        self.provider = provider
        self.retriever = retriever
        self.context_builder = RAGContextBuilder(
            max_articles=max_articles
        )

    def answer(self, complaint: str) -> str:
        retrieved = self.retriever.search(
            complaint,
            top_k=3,
        )

        context = self.context_builder.build_context(
            complaint,
            retrieved,
        )

        prompt = f"""
You are a telecom support resolution assistant.

Use the knowledge-base context below to help resolve the customer's complaint.

IMPORTANT RULES:
- Use only information supported by the knowledge base.
- Do not invent troubleshooting steps, policies, causes, or technical values.
- If the knowledge base does not contain enough information, clearly say so.
- Do not claim that an action was performed.
- Give practical troubleshooting guidance in a clear order.
- Mention escalation conditions when relevant.
- Keep the response concise and suitable for a support agent.

CUSTOMER COMPLAINT:
{complaint}

KNOWLEDGE BASE CONTEXT:
{context}

Provide:
1. Likely issue
2. Recommended troubleshooting steps
3. Escalation condition, if applicable
4. Short agent-facing response
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

        return response.text