from app.llm.factory import get_provider
from app.services.rag_answer import RAGAnswerService
from app.services.retriever import HybridRetriever


def test_rag_answer():
    provider = get_provider()
    retriever = HybridRetriever()

    service = RAGAnswerService(
        provider=provider,
        retriever=retriever,
        max_articles=3,
    )

    answer = service.answer(
        "My internet keeps dropping every evening around 8 PM. "
        "I already restarted the router twice."
    )

    assert answer
    assert len(answer.strip()) > 20