from typing import List


class RAGContextBuilder:
    def __init__(self, max_articles: int = 3):
        self.max_articles = max_articles

    def build_context(
        self,
        complaint: str,
        retrieved_articles: List[dict],
    ) -> str:
        articles = retrieved_articles[:self.max_articles]

        if not articles:
            return "No relevant knowledge-base articles were retrieved."

        sections = []

        for article in articles:
            steps = "\n".join(
                f"  {step['n']}. {step['text']}"
                for step in article.get("steps", [])
            )

            questions = "\n".join(
                f"  - {q['q']}"
                for q in article.get("diagnostic_questions", [])
            )

            escalation = "\n".join(
                f"  - {item}"
                for item in article.get("escalate_when", [])
            )

            section = f"""
KB ARTICLE {article['rank']}
ID: {article['id']}
Title: {article['title']}
Class: {article['class_id']}
Product: {article['product']}
Root cause: {article['root_cause']}

Symptoms:
{chr(10).join(f"  - {s}" for s in article.get("symptoms", []))}

Diagnostic questions:
{questions}

Troubleshooting steps:
{steps}

Escalate when:
{escalation}

Notes:
{article.get("notes", "")}
""".strip()

            sections.append(section)

        return "\n\n" + "\n\n".join(
            "=" * 60 + "\n" + section
            for section in sections
        )