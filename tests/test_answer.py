from app.services.answer import RAGAnswerService
from app.services.retriever import HybridRetriever
from app.llm.mock import MockProvider


def test_rag_answer_contains_retrieval_evidence():
    retriever = HybridRetriever()

    service = RAGAnswerService(
        retriever=retriever,
        provider=MockProvider(),
    )

    result = service.answer(
        "My internet keeps dropping every evening around 8 pm"
    )

    assert result is not None
    assert "retrieved_articles" in result
    assert len(result["retrieved_articles"]) > 0