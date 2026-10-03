from app.db.database import Base, engine
from app.db.models import KBArticleDB


def main():
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")


if __name__ == "__main__":
    main()