from app.services.retriever import HybridRetriever


def test_retriever_returns_results():
    retriever = HybridRetriever()

    results = retriever.search(
        "My internet keeps dropping every evening"
    )

    assert results
    assert len(results) > 0


def test_retriever_returns_relevant_result():
    retriever = HybridRetriever()

    results = retriever.search(
        "My internet keeps dropping every evening"
    )

    assert any(
        result["class_id"] == "connectivity.intermittent"
        for result in results
    )


def test_retriever_returns_top_k_results():
    retriever = HybridRetriever()

    results = retriever.search(
        "My internet keeps dropping every evening",
        top_k=3,
    )

    assert len(results) == 3


def test_retriever_results_have_required_fields():
    retriever = HybridRetriever()

    results = retriever.search(
        "My internet keeps dropping every evening"
    )

    required_fields = {
        "rank",
        "id",
        "title",
        "class_id",
        "product",
        "root_cause",
        "symptoms",
        "steps",
        "escalate_when",
        "notes",
        "dense_score",
        "bm25_score",
    }

    assert required_fields.issubset(results[0].keys())