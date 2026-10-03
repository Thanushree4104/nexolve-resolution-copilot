import json
from pathlib import Path

from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import KBArticleDB


KB_PATH = Path("data/synthetic/kb_articles.jsonl")


def load_articles():
    articles = []

    for line in KB_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            articles.append(json.loads(line))

    return articles


def import_kb():
    articles = load_articles()

    with SessionLocal() as db:
        inserted = 0
        skipped = 0

        for article in articles:
            existing = db.scalar(
                select(KBArticleDB).where(
                    KBArticleDB.id == article["id"]
                )
            )

            if existing:
                skipped += 1
                continue

            db.add(
                KBArticleDB(
                    id=article["id"],
                    title=article["title"],
                    class_id=article["class_id"],
                    product=article["product"],
                    root_cause=article["root_cause"],
                    symptoms=article["symptoms"],
                    diagnostic_questions=article[
                        "diagnostic_questions"
                    ],
                    steps=article["steps"],
                    escalate_when=article["escalate_when"],
                    notes=article.get("notes", ""),
                    version=article.get("version", 1),
                    last_reviewed=article["last_reviewed"],
                    status=article.get("status", "active"),
                    synthetic=article.get("synthetic", True),
                )
            )

            inserted += 1

        db.commit()

    print(f"Imported: {inserted}")
    print(f"Skipped: {skipped}")
    print(f"Total source articles: {len(articles)}")


if __name__ == "__main__":
    import_kb()