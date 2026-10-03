import re
import unicodedata


class RAGContextBuilder:
    def __init__(self, max_articles=3):
        self.max_articles = max_articles

    def build_context(self, complaint, articles):
        articles = articles[: self.max_articles]

        if not articles:
            return "No relevant knowledge-base articles were retrieved."

        sections = []

        for article in articles:
            sections.append(
                f"""
KNOWLEDGE BASE ARTICLE
ID: {article["id"]}
Title: {article["title"]}
Class: {article["class_id"]}
Product: {article["product"]}
Root cause: {article["root_cause"]}

Symptoms:
{self._format_list(article["symptoms"])}

Diagnostic questions:
{self._format_questions(article["diagnostic_questions"])}

Troubleshooting steps:
{self._format_steps(article["steps"])}

Escalation conditions:
{self._format_list(article["escalate_when"])}

Notes:
{article["notes"]}
""".strip()
            )

        return "\n\n---\n\n".join(sections)

    @staticmethod
    def _format_list(items):
        if not items:
            return "- None provided"

        return "\n".join(f"- {item}" for item in items)

    @staticmethod
    def _format_questions(questions):
        if not questions:
            return "- None provided"

        lines = []

        for q in questions:
            lines.append(
                f"- Question: {q['q']}\n"
                f"  If yes: {q['if_yes']}\n"
                f"  If no: {q['if_no']}"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_steps(steps):
        if not steps:
            return "- None provided"

        return "\n".join(
            f"- Step {step['n']}: {step['text']}"
            for step in steps
        )


def clean_llm_output(text: str) -> str:
    """
    Clean common Unicode/mojibake issues from LLM output.
    Keeps the response readable in Windows PowerShell.
    """

    if not text:
        return ""

    # Try to repair common UTF-8 interpreted as Latin-1/Windows-1252.
    if any(x in text for x in ("â", "Ã", "Â", "ð")):
        try:
            repaired = text.encode("latin1").decode("utf-8")
            text = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    # Normalize Unicode.
    text = unicodedata.normalize("NFKC", text)

    # Replace common typographic characters.
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

    # Remove remaining control characters except newline/tab.
    text = "".join(
        char
        for char in text
        if char in "\n\t" or ord(char) >= 32
    )

    # Clean excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


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
            top_k=self.context_builder.max_articles,
        )

        context = self.context_builder.build_context(
            complaint,
            retrieved,
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

        return clean_llm_output(response.text)