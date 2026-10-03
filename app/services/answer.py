from typing import Any

from app.llm.base import LLMError
from app.services.retriever import HybridRetriever


ANSWER_PROMPT = """You are a telecom support resolution assistant.

Your job is to provide a resolution for the customer's complaint using ONLY
the retrieved knowledge-base articles provided below.

IMPORTANT RULES:
- Do not invent troubleshooting steps.
- Do not use knowledge outside the retrieved articles.
- Prefer information from the highest-ranked relevant article.
- If the retrieved articles do not contain enough information, say so clearly.
- Do not claim that an action was performed.
- Give the support agent practical next steps.
- Preserve important escalation conditions.
- Keep the response concise and operational.

CUSTOMER COMPLAINT:
{complaint}

RETRIEVED KNOWLEDGE BASE ARTICLES:
{context}

Return JSON only:

{{
  "summary": "short description of the likely issue",
  "likely_root_cause": "root cause supported by the retrieved articles",
  "recommended_steps": [
    "step 1",
    "step 2"
  ],
  "escalate_when": [
    "condition 1"
  ],
  "confidence": 0.0,
  "grounded": true
}}
"""


def format_context(results: list[dict[str, Any]]) -> str:
    sections = []

    for result in results:
        sections.append(
            f"""
ARTICLE {result["rank"]}
ID: {result["id"]}
TITLE: {result["title"]}
CLASS: {result["class_id"]}
PRODUCT: {result["product"]}
ROOT CAUSE: {result["root_cause"]}

SYMPTOMS:
{chr(10).join("- " + s for s in result["symptoms"])}

DIAGNOSTIC QUESTIONS:
{chr(10).join(
    "- " + q["q"] + " | YES: " + q["if_yes"] + " | NO: " + q["if_no"]
    for q in result["diagnostic_questions"]
)}

STEPS:
{chr(10).join(
    f'{s["n"]}. {s["text"]}'
    for s in result["steps"]
)}

ESCALATE WHEN:
{chr(10).join("- " + x for x in result["escalate_when"])}

NOTES:
{result["notes"]}
"""
        )

    return "\n".join(sections)


class RAGAnswerService:
    def __init__(self, retriever: HybridRetriever, provider):
        self.retriever = retriever
        self.provider = provider

    def answer(self, complaint: str, top_k: int = 5) -> dict:
        results = self.retriever.search(
            complaint,
            top_k=top_k,
        )

        if not results:
            return {
                "summary": "No relevant knowledge-base article was found.",
                "likely_root_cause": None,
                "recommended_steps": [],
                "escalate_when": [],
                "confidence": 0.0,
                "grounded": False,
                "retrieved_articles": [],
            }

        context = format_context(results)

        prompt = ANSWER_PROMPT.format(
            complaint=complaint,
            context=context,
        )

        response = self.provider.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1200,
            json_mode=True,
        )

        try:
            import json

            start = response.text.find("{")
            end = response.text.rfind("}")

            if start == -1 or end == -1:
                raise ValueError("No JSON object found")

            answer = json.loads(response.text[start : end + 1])

        except Exception as exc:
            raise LLMError(
                f"Invalid RAG answer returned by LLM: {str(exc)[:200]}"
            ) from exc

        answer["retrieved_articles"] = [
            {
                "rank": r["rank"],
                "id": r["id"],
                "title": r["title"],
                "class_id": r["class_id"],
                "root_cause": r["root_cause"],
                "dense_score": r["dense_score"],
                "bm25_score": r["bm25_score"],
            }
            for r in results
        ]

        return answer