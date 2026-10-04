from app.services.feedback_reranker import (
    FeedbackStore,
    FeedbackReranker,
)


def main():
    store = FeedbackStore()

    reranker = FeedbackReranker(
        feedback_store=store,
        feedback_weight=0.10,
    )

    articles = [
        {
            "id": "KB-1001",
            "dense_score": 0.80,
            "bm25_score": 0.70,
        },
        {
            "id": "KB-1002",
            "dense_score": 0.78,
            "bm25_score": 0.72,
        },
        {
            "id": "KB-1003",
            "dense_score": 0.75,
            "bm25_score": 0.71,
        },
    ]

    store.add_feedback(
        query="internet keeps disconnecting",
        kb_id="KB-1003",
        feedback=1,
    )

    store.add_feedback(
        query="internet keeps disconnecting",
        kb_id="KB-1001",
        feedback=-1,
    )

    results = reranker.rerank(
        query="internet keeps disconnecting",
        articles=articles,
    )

    print("=" * 70)
    print("FEEDBACK RERANKING TEST")
    print("=" * 70)

    for article in results:
        print(
            article["rank"],
            article["id"],
            "base=",
            round(
                reranker._base_score(article),
                4,
            ),
            "feedback=",
            article["feedback_score"],
            "final=",
            round(
                article["rerank_score"],
                4,
            ),
        )


if __name__ == "__main__":
    main()