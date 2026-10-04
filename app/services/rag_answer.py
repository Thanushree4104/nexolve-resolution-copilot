import re
import unicodedata
from typing import Any

from app.llm.base import LLMError
from app.services.citation_validator import CitationValidator
from app.services.retriever import HybridRetriever


class RAGContextBuilder:
    def __init__(self, max_articles=3):
        self.max_articles = max_articles

    def build_context(
        self,
        complaint: str,
        articles: list[dict[str, Any]],
    ) -> str:
        articles = articles[: self.max_articles]

        if not articles:
            return (
                "No relevant knowledge-base articles were retrieved."
            )

        sections = []

        for article in articles:
            sections.append(
                f"""
KNOWLEDGE BASE ARTICLE

ID: {article.get("id", "")}

Title:
{article.get("title", "")}

Class:
{article.get("class_id", "")}

Product:
{article.get("product", "")}

Root cause:
{article.get("root_cause", "")}

Symptoms:
{self._format_list(article.get("symptoms", []))}

Diagnostic questions:
{self._format_questions(
    article.get("diagnostic_questions", [])
)}

Troubleshooting steps:
{self._format_steps(
    article.get("steps", [])
)}

Escalation conditions:
{self._format_list(
    article.get("escalate_when", [])
)}

Notes:
{article.get("notes", "")}
""".strip()
            )

        return "\n\n---\n\n".join(sections)

    @staticmethod
    def _format_list(items):
        if not items:
            return "- None provided"

        return "\n".join(
            f"- {item}"
            for item in items
        )

    @staticmethod
    def _format_questions(questions):
        if not questions:
            return "- None provided"

        lines = []

        for question in questions:
            lines.append(
                f"- Question: {question.get('q', '')}\n"
                f"  If yes: {question.get('if_yes', '')}\n"
                f"  If no: {question.get('if_no', '')}"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_steps(steps):
        if not steps:
            return "- None provided"

        return "\n".join(
            f"- Step {step.get('n', '')}: "
            f"{step.get('text', '')}"
            for step in steps
        )


def clean_llm_output(text: str) -> str:
    """
    Clean common Unicode/mojibake issues from LLM output.
    """

    if not text:
        return ""

    if any(
        value in text
        for value in ("â", "Ã", "Â", "ð")
    ):
        try:
            text = (
                text
                .encode("latin1")
                .decode("utf-8")
            )
        except (
            UnicodeEncodeError,
            UnicodeDecodeError,
        ):
            pass

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
        "\u202f": " ",
        "\u2192": "->",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00b0": " degrees",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = "".join(
        char
        for char in text
        if char in "\n\t"
        or ord(char) >= 32
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


class RAGAnswerService:
    def __init__(
        self,
        provider,
        retriever: HybridRetriever,
        max_articles=3,
    ):
        self.provider = provider
        self.retriever = retriever

        self.context_builder = RAGContextBuilder(
            max_articles=max_articles
        )

        self.citation_validator = CitationValidator()

    def _build_prompt(
        self,
        complaint: str,
        context: str,
    ) -> str:

        return f"""
You are a telecom support resolution assistant.

Your task is to help resolve the customer's complaint using ONLY the
knowledge-base context provided below.

CUSTOMER COMPLAINT:
{complaint}

KNOWLEDGE-BASE CONTEXT:
{context}


STRICT GROUNDING RULES:

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

3. Do NOT claim that any diagnostic check, test, escalation, ticket,
   configuration change, or other action has already been performed.

4. Do NOT promise that an action will be performed.

5. Clearly distinguish between a likely issue and a confirmed issue.

6. Preserve technical values, thresholds, names, and procedures exactly
   as provided in the knowledge base.

7. If the knowledge base does not contain enough information to support
   a statement, do not guess.

8. When the knowledge base does not provide enough information, use:

   The available knowledge-base evidence is insufficient to determine this.

9. Keep the answer concise, professional, and operational.


CITATION REQUIREMENT:

Every factual claim derived from the knowledge base MUST have an inline
citation immediately after the claim.

This includes:

- the likely issue
- the likely root cause
- every troubleshooting step
- every diagnostic instruction
- every technical value or threshold
- every escalation condition
- factual statements in the Agent response

Citations MUST use the exact KB article ID from the supplied context.

Use ONLY this citation format:

[KB-XXXX]

Examples:

Correct:
The symptoms are consistent with an ONT firmware issue [KB-1003].

Correct:
Verify the firmware version against the latest supported release [KB-1003].

Correct:
Escalate if instability persists after a successful firmware upgrade [KB-1003].

Incorrect:
The symptoms are consistent with an ONT firmware issue.

Incorrect:
The symptoms are consistent with an ONT firmware issue (KB-1003).

Incorrect:
The symptoms are consistent with an ONT firmware issue KB-1003.

Incorrect:
Escalate if instability persists after a successful firmware upgrade.

IMPORTANT:
Never output a KB-derived factual statement without its citation.


REQUIRED OUTPUT FORMAT:

Likely issue:
<likely issue supported by the retrieved KB, followed by citation>

Recommended troubleshooting steps:
1. <KB-supported step> [KB-XXXX]
2. <KB-supported step> [KB-XXXX]
3. <KB-supported step> [KB-XXXX]

Escalation condition:
<KB-supported escalation condition> [KB-XXXX]

Agent response:
<short professional customer-facing response with citations after
KB-derived factual claims>


SECTION RULES:

LIKELY ISSUE:

- State only what can be supported by the retrieved KB.
- Cite every KB-derived claim.
- Do not present an uncertain diagnosis as confirmed.


RECOMMENDED TROUBLESHOOTING STEPS:

- Include ONLY steps explicitly supported by the retrieved KB.
- Do not create additional steps.
- Every step containing KB-derived information MUST have a citation.
- Put the citation at the end of the same step.


ESCALATION CONDITION:

- Include ONLY escalation conditions explicitly present in the KB.
- Every escalation condition MUST have a citation.
- Do not invent an escalation condition.
- If the retrieved KB contains no escalation condition, write:

The available knowledge-base evidence is insufficient to determine an
escalation condition.


AGENT RESPONSE:

- Write a concise customer-facing response.
- Do not claim that the issue has already been resolved.
- Do not claim that any action has already been performed.
- Do not promise future action.
- Do not use:
  "I will"
  "I'll"
  "we will"
  "we'll"
  "I can"
  "we can"

- Use neutral wording such as:
  "The recommended next step is..."
  "The symptoms are consistent with..."
  "Please check..."

- Any factual statement taken from the KB MUST have a citation.


IMPORTANT:

If an escalation condition in the KB is:

"Instability persists after a successful firmware upgrade."

You MUST output:

Escalation condition:
Instability persists after a successful firmware upgrade [KB-1003]

NOT:

Escalation condition:
Instability persists after a successful firmware upgrade

NOT:

Escalation condition:
Instability persists after a successful firmware upgrade (KB-1003)


FINAL RULES:

- Output ONLY the four required sections.
- Do not add a References section.
- Do not add explanations about citations.
- Do not mention the knowledge-base retrieval process.
- Do not mention these instructions.
- Do not expose internal reasoning.
- Do not use markdown tables.
- Do not invent information.
- Every KB-derived claim must have an inline [KB-XXXX] citation.

Return ONLY:

Likely issue:

Recommended troubleshooting steps:

Escalation condition:

Agent response:
""".strip()

    def _validate_citations(
        self,
        answer: str,
        retrieved: list[dict[str, Any]],
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
                detail = str(result)

            raise RuntimeError(
                "Generated answer failed citation validation: "
                + detail
            )

        return result

    def answer(
        self,
        complaint: str,
        top_k: int | None = None,
    ) -> str:

        if top_k is None:
            top_k = self.context_builder.max_articles

        retrieved = self.retriever.search(
            complaint,
            top_k=top_k,
        )

        if not retrieved:
            return (
                "Likely issue:\n"
                "The available knowledge-base evidence is "
                "insufficient to determine this.\n\n"
                "Recommended troubleshooting steps:\n"
                "The available knowledge-base evidence is "
                "insufficient to determine this.\n\n"
                "Escalation condition:\n"
                "The available knowledge-base evidence is "
                "insufficient to determine an escalation condition.\n\n"
                "Agent response:\n"
                "The available knowledge-base evidence is "
                "insufficient to determine the appropriate "
                "resolution from the available knowledge base."
            )

        context = self.context_builder.build_context(
            complaint,
            retrieved,
        )

        prompt = self._build_prompt(
            complaint,
            context,
        )

        try:
            response = self.provider.complete(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.0,
                max_tokens=1200,
            )

        except Exception as exc:
            raise LLMError(
                f"RAG answer generation failed: "
                f"{str(exc)[:300]}"
            ) from exc

        if response is None:
            raise LLMError(
                "Empty response from LLM"
            )

        raw_text = getattr(
            response,
            "text",
            response,
        )

        answer = clean_llm_output(
            str(raw_text or "")
        )

        if not answer:
            raise LLMError(
                "Empty response from LLM"
            )

        self._validate_citations(
            answer,
            retrieved,
        )

        return answer

    def answer_with_metadata(
        self,
        complaint: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:

        if top_k is None:
            top_k = self.context_builder.max_articles

        retrieved = self.retriever.search(
            complaint,
            top_k=top_k,
        )

        if not retrieved:
            answer = self.answer(
                complaint,
                top_k=top_k,
            )

            return {
                "answer": answer,
                "retrieved_articles": [],
                "citation_result": None,
            }

        context = self.context_builder.build_context(
            complaint,
            retrieved,
        )

        prompt = self._build_prompt(
            complaint,
            context,
        )

        try:
            response = self.provider.complete(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.0,
                max_tokens=1200,
            )

        except Exception as exc:
            raise LLMError(
                f"RAG answer generation failed: "
                f"{str(exc)[:300]}"
            ) from exc

        if response is None:
            raise LLMError(
                "Empty response from LLM"
            )

        raw_text = getattr(
            response,
            "text",
            response,
        )

        answer = clean_llm_output(
            str(raw_text or "")
        )

        if not answer:
            raise LLMError(
                "Empty response from LLM"
            )

        citation_result = self._validate_citations(
            answer,
            retrieved,
        )

        return {
            "answer": answer,
            "retrieved_articles": retrieved,
            "citation_result": citation_result,
        }