from app.services.rag import RAGContextBuilder


def test_rag_context_contains_retrieved_article():
    builder = RAGContextBuilder(max_articles=2)

    articles = [
        {
            "rank": 1,
            "id": "KB-1001",
            "title": "Evening Node Load Leads to Unstable Nimbus Fibre Service",
            "class_id": "connectivity.intermittent",
            "product": "Nimbus Fibre",
            "root_cause": "evening congestion on the shared node",
            "symptoms": [
                "Websites become unresponsive after a few minutes"
            ],
            "diagnostic_questions": [
                {
                    "q": "Is the issue observed primarily after 6 pm?",
                    "if_yes": "Check node utilization",
                    "if_no": "Investigate alternatives",
                }
            ],
            "steps": [
                {
                    "n": 1,
                    "text": "Check the shared node."
                }
            ],
            "escalate_when": [
                "Node utilization remains above 85%"
            ],
            "notes": "Document timestamps.",
            "dense_score": 0.85,
            "bm25_score": 3.2,
        }
    ]

    context = builder.build_context(
        "My internet keeps dropping every evening",
        articles,
    )

    assert "KB-1001" in context
    assert "Evening Node Load" in context
    assert "evening congestion" in context
    assert "Check the shared node" in context