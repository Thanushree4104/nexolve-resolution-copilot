from app.services.retriever import HybridRetriever


def main():
    complaint = (
        "My internet keeps dropping every evening around 8 PM "
        "and restarting the router does not help."
    )

    retriever = HybridRetriever()

    results = retriever.search(
        complaint,
        top_k=5,
    )

    print("\nHybrid Retrieval Results\n")
    print(
        f"{'Rank':<6}"
        f"{'ID':<10}"
        f"{'Dense':<10}"
        f"{'BM25':<10}"
        f"Title"
    )

    print("-" * 100)

    for result in results:
        print(
            f"{result['rank']:<6}"
            f"{result['id']:<10}"
            f"{result['dense_score']:<10.4f}"
            f"{result['bm25_score']:<10.4f}"
            f"{result['title']}"
        )


if __name__ == "__main__":
    main()