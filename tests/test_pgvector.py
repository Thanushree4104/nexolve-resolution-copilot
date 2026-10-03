from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from app.db.database import engine


MODEL = "intfloat/multilingual-e5-small"

complaint = (
    "My internet keeps dropping every evening around 8 PM "
    "and restarting the router does not help."
)


def main():
    model = SentenceTransformer(MODEL)

    query_embedding = model.encode(
        ["query: " + complaint],
        normalize_embeddings=True,
    )[0].tolist()

    sql = text("""
        SELECT
            id,
            title,
            class_id,
            root_cause,
            1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM kb_articles
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT 5
    """)

    with engine.connect() as connection:
        rows = connection.execute(
            sql,
            {"embedding": str(query_embedding)},
        ).fetchall()

    for row in rows:
        print(
            f"{row.id} | "
            f"{row.class_id} | "
            f"{row.similarity:.4f} | "
            f"{row.title}"
        )


if __name__ == "__main__":
    main()