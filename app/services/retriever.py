import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from app.core.pii import redact
from app.db.database import engine


MODEL = "intfloat/multilingual-e5-small"
RRF_K = 60


class HybridRetriever:
    def __init__(
        self,
        model_name=MODEL,
        candidate_k=10,
    ):
        self.model_name = model_name
        self.candidate_k = candidate_k

        self.model = SentenceTransformer(
            self.model_name
        )

        self.articles = self._load_articles()

        if not self.articles:
            raise ValueError(
                "Knowledge base contains no articles"
            )

        self.article_by_id = {
            article["id"]: article
            for article in self.articles
        }

        self.doc_texts = [
            self._doc_text(article)
            for article in self.articles
        ]

        # Keep BM25 as our existing lexical baseline.
        self.bm25 = BM25Okapi(
            [
                self._tokenize(text)
                for text in self.doc_texts
            ]
        )

    def _load_articles(self):
        """
        Load KB metadata for BM25 and result formatting.

        Dense embeddings are no longer loaded into memory.
        They are stored and searched in PostgreSQL/pgvector.
        """

        import json
        from pathlib import Path

        kb_path = Path(
            "data/synthetic/kb_articles.jsonl"
        )

        return [
            json.loads(line)
            for line in kb_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

    @staticmethod
    def _doc_text(article):
        """
        Build the same retrieval representation used
        by the previous BM25 implementation.
        """

        title = article.get(
            "title",
            "",
        )

        class_id = article.get(
            "class_id",
            "",
        )

        product = article.get(
            "product",
            "",
        )

        root_cause = article.get(
            "root_cause",
            "",
        )

        symptoms = " ".join(
            article.get(
                "symptoms",
                [],
            )
        )

        questions = article.get(
            "diagnostic_questions",
            [],
        )

        diagnostic_questions = " ".join(
            question.get(
                "q",
                "",
            )
            for question in questions
        )

        diagnostic_guidance = " ".join(
            (
                question.get(
                    "if_yes",
                    "",
                )
                + " "
                + question.get(
                    "if_no",
                    "",
                )
            )
            for question in questions
        )

        return (
            f"Title: {title}. "
            f"Problem class: {class_id}. "
            f"Product: {product}. "
            f"Root cause: {root_cause}. "
            f"Symptoms: {symptoms}. "
            f"Diagnostic questions: "
            f"{diagnostic_questions}. "
            f"Diagnostic guidance: "
            f"{diagnostic_guidance}."
        )

    @staticmethod
    def _tokenize(text):
        return re.findall(
            r"\w+",
            text.lower(),
            flags=re.UNICODE,
        )

    @staticmethod
    def _rrf(
        dense_order,
        bm25_order,
        k=RRF_K,
    ):
        scores = {}

        for rank, article_id in enumerate(
            dense_order
        ):
            scores[article_id] = (
                scores.get(article_id, 0.0)
                + 1.0 / (k + rank + 1)
            )

        for rank, article_id in enumerate(
            bm25_order
        ):
            scores[article_id] = (
                scores.get(article_id, 0.0)
                + 1.0 / (k + rank + 1)
            )

        return sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

    def _pgvector_search(
        self,
        query_vector,
        limit,
    ):
        """
        Semantic retrieval using PostgreSQL + pgvector.

        The query uses cosine distance:
            embedding <=> query_embedding

        Because embeddings are normalized, this is equivalent
        to cosine similarity ranking.
        """

        sql = text(
            """
            SELECT
                id,
                1 - (
                    embedding
                    <=> CAST(:embedding AS vector)
                ) AS dense_score
            FROM kb_articles
            WHERE embedding IS NOT NULL
              AND status = 'active'
            ORDER BY embedding
                <=> CAST(:embedding AS vector)
            LIMIT :limit
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(
                sql,
                {
                    "embedding": str(
                        query_vector.tolist()
                    ),
                    "limit": limit,
                },
            ).mappings().all()

        return rows

    def search(
        self,
        complaint,
        top_k=5,
    ):
        """
        Hybrid retrieval:

        1. Redact PII.
        2. Generate query embedding.
        3. Semantic search through pgvector.
        4. BM25 lexical search in Python.
        5. Fuse rankings using RRF.
        6. Return the same result structure expected
           by the existing RAG pipeline.
        """

        clean_complaint = redact(
            complaint
        ).text

        # -------------------------------------------------
        # Dense semantic retrieval
        # -------------------------------------------------

        query_vector = self.model.encode(
            [
                "query: " + clean_complaint
            ],
            normalize_embeddings=True,
        )[0]

        candidate_k = max(
            self.candidate_k,
            top_k,
        )

        dense_rows = self._pgvector_search(
            query_vector,
            candidate_k,
        )

        dense_order = [
            row["id"]
            for row in dense_rows
        ]

        dense_scores = {
            row["id"]: float(
                row["dense_score"]
            )
            for row in dense_rows
        }

        # -------------------------------------------------
        # BM25 lexical retrieval
        # -------------------------------------------------

        query_tokens = self._tokenize(
            clean_complaint
        )

        bm25_scores_array = self.bm25.get_scores(
            query_tokens
        )

        bm25_indices = np.argsort(
            -bm25_scores_array
        )[:candidate_k]

        bm25_order = [
            self.articles[index]["id"]
            for index in bm25_indices
        ]

        bm25_scores = {
            self.articles[index]["id"]: float(
                bm25_scores_array[index]
            )
            for index in bm25_indices
        }

        # -------------------------------------------------
        # Reciprocal Rank Fusion
        # -------------------------------------------------

        hybrid_order = self._rrf(
            dense_order,
            bm25_order,
        )

        # -------------------------------------------------
        # Build final results
        # -------------------------------------------------

        results = []

        for rank, article_id in enumerate(
            hybrid_order[:top_k],
            start=1,
        ):
            article = self.article_by_id.get(
                article_id
            )

            if not article:
                continue

            results.append(
                {
                    "rank": rank,
                    "id": article["id"],
                    "title": article["title"],
                    "class_id": article["class_id"],
                    "product": article["product"],
                    "root_cause": article[
                        "root_cause"
                    ],
                    "symptoms": article[
                        "symptoms"
                    ],
                    "diagnostic_questions": article[
                        "diagnostic_questions"
                    ],
                    "steps": article[
                        "steps"
                    ],
                    "escalate_when": article[
                        "escalate_when"
                    ],
                    "notes": article[
                        "notes"
                    ],
                    "dense_score": dense_scores.get(
                        article_id,
                        0.0,
                    ),
                    "bm25_score": bm25_scores.get(
                        article_id,
                        0.0,
                    ),
                }
            )

        return results