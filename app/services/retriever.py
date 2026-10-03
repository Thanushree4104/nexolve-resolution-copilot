import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.pii import redact


MODEL = "intfloat/multilingual-e5-small"
RRF_K = 60


class HybridRetriever:
    def __init__(
        self,
        kb_path="data/synthetic/kb_articles.jsonl",
        model_name=MODEL,
    ):
        self.kb_path = Path(kb_path)
        self.model_name = model_name

        self.articles = self._load_articles()

        if not self.articles:
            raise ValueError(
                "Knowledge base contains no articles"
            )

        self.ids = [
            article["id"]
            for article in self.articles
        ]

        self.model = SentenceTransformer(
            self.model_name
        )

        self.doc_texts = [
            self._doc_text(article)
            for article in self.articles
        ]

        self.doc_vectors = self.model.encode(
            [
                "passage: " + text
                for text in self.doc_texts
            ],
            normalize_embeddings=True,
            batch_size=32,
        )

        self.bm25 = BM25Okapi(
            [
                self._tokenize(text)
                for text in self.doc_texts
            ]
        )

    def _load_articles(self):
        return [
            json.loads(line)
            for line in self.kb_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

    @staticmethod
    def _doc_text(article):
        """
        Build a retrieval-focused representation.

        Includes:
        - title
        - class
        - product
        - root cause
        - symptoms
        - diagnostic questions
        - diagnostic guidance

        The diagnostic information helps distinguish
        KB articles that have similar symptoms but
        different underlying causes.
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
        """
        Tokenize text for BM25.
        """

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
        """
        Reciprocal Rank Fusion.

        Combines the rankings produced by
        dense similarity search and BM25.
        """

        scores = {}

        for rank, index in enumerate(
            dense_order
        ):
            scores[index] = (
                scores.get(index, 0.0)
                + 1.0 / (k + rank + 1)
            )

        for rank, index in enumerate(
            bm25_order
        ):
            scores[index] = (
                scores.get(index, 0.0)
                + 1.0 / (k + rank + 1)
            )

        return sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

    def search(
        self,
        complaint,
        top_k=5,
    ):
        """
        Retrieve the most relevant KB articles
        using hybrid dense + BM25 retrieval.
        """

        clean_complaint = redact(
            complaint
        ).text

        # -------------------------
        # Dense semantic retrieval
        # -------------------------

        query_vector = self.model.encode(
            [
                "query: " + clean_complaint
            ],
            normalize_embeddings=True,
        )[0]

        dense_scores = (
            query_vector
            @ self.doc_vectors.T
        )

        dense_order = np.argsort(
            -dense_scores
        )

        # -------------------------
        # BM25 keyword retrieval
        # -------------------------

        query_tokens = self._tokenize(
            clean_complaint
        )

        bm25_scores = self.bm25.get_scores(
            query_tokens
        )

        bm25_order = np.argsort(
            -bm25_scores
        )

        # -------------------------
        # Hybrid retrieval
        # -------------------------

        hybrid_order = self._rrf(
            dense_order,
            bm25_order,
        )

        # -------------------------
        # Build results
        # -------------------------

        results = []

        for rank, index in enumerate(
            hybrid_order[:top_k],
            start=1,
        ):
            article = self.articles[index]

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
                    "dense_score": float(
                        dense_scores[index]
                    ),
                    "bm25_score": float(
                        bm25_scores[index]
                    ),
                }
            )

        return results