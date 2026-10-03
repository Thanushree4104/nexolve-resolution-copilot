from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.db.database import Base


EMBEDDING_DIM = 384


class KBArticleDB(Base):
    __tablename__ = "kb_articles"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    class_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    product: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    root_cause: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    symptoms: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
    )

    diagnostic_questions: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
    )

    steps: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
    )

    escalate_when: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
    )

    notes: Mapped[str] = mapped_column(
        Text,
        default="",
        nullable=False,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    last_reviewed: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="active",
        nullable=False,
        index=True,
    )

    synthetic: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    embedding: Mapped[list | None] = mapped_column(
        Vector(EMBEDDING_DIM),
        nullable=True,
    )