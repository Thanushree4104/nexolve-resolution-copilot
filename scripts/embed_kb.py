from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import KBArticleDB


MODEL = "intfloat/multilingual-e5-small"


def build_doc_text(article):
    symptoms = " ".join(article.symptoms)

    questions = article.diagnostic_questions

    diagnostic_questions = " ".join(
        question.get("q", "")
        for question in questions
    )

    diagnostic_guidance = " ".join(
        question.get("if_yes", "")
        + " "
        + question.get("if_no", "")
        for question in questions
    )

    return (
        f"Title: {article.title}. "
        f"Problem class: {article.class_id}. "
        f"Product: {article.product}. "
        f"Root cause: {article.root_cause}. "
        f"Symptoms: {symptoms}. "
        f"Diagnostic questions: {diagnostic_questions}. "
        f"Diagnostic guidance: {diagnostic_guidance}."
    )


def main():
    model = SentenceTransformer(MODEL)

    with SessionLocal() as db:
        articles = db.scalars(
            select(KBArticleDB)
        ).all()

        texts = [
            "passage: " + build_doc_text(article)
            for article in articles
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
        )

        for article, embedding in zip(
            articles,
            embeddings,
        ):
            article.embedding = embedding.tolist()

        db.commit()

    print(f"Embedded {len(articles)} KB articles.")


if __name__ == "__main__":
    main()