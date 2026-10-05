import json
import logging
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.services.feedback_reranker import FeedbackReranker
from app.core.pii import redact


logger = logging.getLogger("app.retrieval")


MODEL = "intfloat/multilingual-e5-small"


# Hybrid fusion weights.
# Dense retrieval is currently the stronger baseline for this dataset,
# so we give it more influence than BM25.
DENSE_WEIGHT = 0.75
BM25_WEIGHT = 0.25


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

        # Feedback-based reranking
        self.feedback_reranker = FeedbackReranker()

        logger.info(
            "retriever_initialized",
            extra={
                "extra_fields": {
                    "retriever": "hybrid",
                    "embedding_model": self.model_name,
                    "kb_articles": len(self.articles),
                    "dense_weight": DENSE_WEIGHT,
                    "bm25_weight": BM25_WEIGHT,
                    "feedback_reranking": True,
                }
            },
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
        title = article.get("title", "")
        class_id = article.get("class_id", "")
        product = article.get("product", "")
        root_cause = article.get("root_cause", "")

        symptoms = " ".join(
            article.get("symptoms", [])
        )

        questions = article.get(
            "diagnostic_questions",
            [],
        )

        diagnostic_questions = " ".join(
            question.get("q", "")
            for question in questions
        )

        diagnostic_guidance = " ".join(
            (
                question.get("if_yes", "")
                + " "
                + question.get("if_no", "")
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
    def _normalize_scores(scores):
        scores = np.asarray(
            scores,
            dtype=float,
        )

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.ones_like(scores)

        return (
            (scores - min_score)
            / (max_score - min_score)
        )

    @classmethod
    def _hybrid_scores(
        cls,
        dense_scores,
        bm25_scores,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
    ):
        """
        Weighted score fusion.

        Dense cosine similarity is normalized to the same
        [0, 1] range as BM25 before combining the two signals.
        """

        dense_scores = np.asarray(
            dense_scores,
            dtype=float,
        )

        bm25_scores = np.asarray(
            bm25_scores,
            dtype=float,
        )

        dense_normalized = cls._normalize_scores(
            dense_scores
        )

        bm25_normalized = cls._normalize_scores(
            bm25_scores
        )

        return (
            dense_weight * dense_normalized
            + bm25_weight * bm25_normalized
        )

    def _score_query(self, complaint):
        """
        Compute all retrieval signals for one query.

        Returns dense scores, BM25 scores, and the corresponding
        rankings. This is shared by production retrieval and
        evaluation so they cannot silently diverge.
        """

        clean_complaint = redact(
            complaint
        ).text

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

        query_tokens = self._tokenize(
            clean_complaint
        )

        bm25_scores = self.bm25.get_scores(
            query_tokens
        )

        bm25_order = np.argsort(
            -bm25_scores
        )

        hybrid_scores = self._hybrid_scores(
            dense_scores,
            bm25_scores,
        )

        hybrid_order = np.argsort(
            -hybrid_scores
        )

        return {
            "clean_query": clean_complaint,
            "dense_scores": dense_scores,
            "bm25_scores": bm25_scores,
            "hybrid_scores": hybrid_scores,
            "dense_order": dense_order,
            "bm25_order": bm25_order,
            "hybrid_order": hybrid_order,
        }

    def search_all(
        self,
        complaint,
        top_k=None,
    ):
        """
        Return dense, BM25, and hybrid rankings for evaluation
        and diagnostics.

        This method does not change the production result format.
        """

        scores = self._score_query(
            complaint
        )

        if top_k is None:
            top_k = len(self.articles)

        return {
            "dense": [
                self.ids[index]
                for index in scores["dense_order"][:top_k]
            ],
            "bm25": [
                self.ids[index]
                for index in scores["bm25_order"][:top_k]
            ],
            "hybrid": [
                self.ids[index]
                for index in scores["hybrid_order"][:top_k]
            ],
        }

    def search(
        self,
        complaint,
        top_k=5,
    ):
        start = time.perf_counter()

        scores = self._score_query(
            complaint
        )

        dense_scores = scores[
            "dense_scores"
        ]

        bm25_scores = scores[
            "bm25_scores"
        ]

        hybrid_scores = scores[
            "hybrid_scores"
        ]

        dense_order = scores[
            "dense_order"
        ]

        bm25_order = scores[
            "bm25_order"
        ]

        hybrid_order = scores[
            "hybrid_order"
        ]

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
                    "steps": article["steps"],
                    "escalate_when": article[
                        "escalate_when"
                    ],
                    "notes": article["notes"],
                    "dense_score": float(
                        dense_scores[index]
                    ),
                    "bm25_score": float(
                        bm25_scores[index]
                    ),
                    "hybrid_score": float(
                        hybrid_scores[index]
                    ),
                }
            )

        # -------------------------------------------------
        # FEEDBACK RERANKING
        # -------------------------------------------------
        # Apply previously collected user feedback to the
        # hybrid retrieval results.
        #
        # The reranker adds:
        #
        #   feedback_score
        #   feedback_normalized
        #   rerank_score
        #
        # and then sorts the articles using rerank_score.
        # -------------------------------------------------

        results = self.feedback_reranker.rerank(
            complaint,
            results,
        )

        # Keep only the requested number of results
        results = results[:top_k]

        # Reassign final ranks after feedback reranking
        for rank, article in enumerate(
            results,
            start=1,
        ):
            article["rank"] = rank

        latency_ms = round(
            (
                time.perf_counter()
                - start
            )
            * 1000,
            1,
        )

        logger.info(
            "retrieval_completed",
            extra={
                "extra_fields": {
                    "retriever": "hybrid",
                    "top_k": top_k,
                    "result_count": len(results),
                    "retrieved_ids": [
                        result["id"]
                        for result in results
                    ],
                    "dense_top_id": self.ids[
                        dense_order[0]
                    ],
                    "bm25_top_id": self.ids[
                        bm25_order[0]
                    ],
                    "hybrid_top_id": self.ids[
                        hybrid_order[0]
                    ],
                    "final_top_id": (
                        results[0]["id"]
                        if results
                        else None
                    ),
                    "latency_ms": latency_ms,
                    "dense_weight": DENSE_WEIGHT,
                    "bm25_weight": BM25_WEIGHT,
                    "feedback_reranking": True,
                }
            },
        )

        return results