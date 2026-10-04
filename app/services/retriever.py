import json
import logging
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.pii import redact


logger = logging.getLogger("app.retrieval")

MODEL = "intfloat/multilingual-e5-small"
RRF_K = 60

# Feedback influence.
# Keep this relatively small so feedback improves ranking
# without completely overriding semantic/BM25 retrieval.
FEEDBACK_WEIGHT = 0.15

# Store feedback separately from the KB.
DEFAULT_FEEDBACK_PATH = "data/eval/retrieval_feedback.jsonl"


class HybridRetriever:
    def __init__(
        self,
        kb_path="data/synthetic/kb_articles.jsonl",
        model_name=MODEL,
        feedback_path=DEFAULT_FEEDBACK_PATH,
    ):
        self.kb_path = Path(kb_path)
        self.model_name = model_name
        self.feedback_path = Path(feedback_path)

        self.articles = self._load_articles()

        if not self.articles:
            raise ValueError(
                "Knowledge base contains no articles"
            )

        self.ids = [
            article["id"]
            for article in self.articles
        ]

        # ---------------------------------------------------------
        # Embedding model
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # BM25
        # ---------------------------------------------------------
        self.bm25 = BM25Okapi(
            [
                self._tokenize(text)
                for text in self.doc_texts
            ]
        )

        # ---------------------------------------------------------
        # Feedback store
        # ---------------------------------------------------------
        self.feedback_store = FeedbackStore(
            self.feedback_path
        )

        logger.info(
            "retriever_initialized",
            extra={
                "extra_fields": {
                    "retriever": "hybrid",
                    "embedding_model": self.model_name,
                    "kb_articles": len(self.articles),
                    "rrf_k": RRF_K,
                    "feedback_path": str(
                        self.feedback_path
                    ),
                }
            },
        )

    # =============================================================
    # KNOWLEDGE BASE
    # =============================================================

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

    # =============================================================
    # RRF
    # =============================================================

    @staticmethod
    def _rrf(
        dense_order,
        bm25_order,
        k=RRF_K,
    ):
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

    # =============================================================
    # FEEDBACK
    # =============================================================

    def record_feedback(
        self,
        query: str,
        kb_id: str,
        positive: bool,
    ) -> None:
        """
        Record whether a KB article was useful for a query.

        positive=True  -> +1
        positive=False -> -1
        """

        if not query or not kb_id:
            return

        if kb_id not in self.ids:
            logger.warning(
                "feedback_unknown_kb_id",
                extra={
                    "extra_fields": {
                        "kb_id": kb_id,
                    }
                },
            )
            return

        feedback = 1 if positive else -1

        self.feedback_store.add_feedback(
            query=query,
            kb_id=kb_id,
            feedback=feedback,
        )

        logger.info(
            "retrieval_feedback_recorded",
            extra={
                "extra_fields": {
                    "kb_id": kb_id,
                    "feedback": feedback,
                }
            },
        )

    def _feedback_scores(self, query):
        """
        Return feedback score for each KB article.

        Scores are normalized to [-1, 1].

        No feedback -> 0.0
        Mostly positive -> positive score
        Mostly negative -> negative score
        """

        scores = {}

        for kb_id in self.ids:
            scores[kb_id] = (
                self.feedback_store.get_score(
                    query=query,
                    kb_id=kb_id,
                )
            )

        return scores

    def _apply_feedback_reranking(
        self,
        hybrid_order,
        query,
    ):
        """
        Apply a small feedback adjustment to the
        existing hybrid ranking.

        The original RRF order remains the primary signal.
        """

        if not hybrid_order:
            return hybrid_order

        feedback_scores = self._feedback_scores(
            query
        )

        # Convert the original ranking into a base score.
        #
        # Higher ranked documents receive a higher base score.
        base_scores = {}

        for rank, index in enumerate(
            hybrid_order
        ):
            base_scores[index] = (
                1.0 / (rank + 1)
            )

        final_scores = {}

        for index in hybrid_order:
            kb_id = self.ids[index]

            feedback = feedback_scores.get(
                kb_id,
                0.0,
            )

            final_scores[index] = (
                base_scores[index]
                + FEEDBACK_WEIGHT * feedback
            )

        return sorted(
            hybrid_order,
            key=lambda index: final_scores[index],
            reverse=True,
        )

    # =============================================================
    # SEARCH
    # =============================================================

    def search(
        self,
        complaint,
        top_k=5,
    ):
        start = time.perf_counter()

        clean_complaint = redact(
            complaint
        ).text

        # ---------------------------------------------------------
        # Dense retrieval
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # BM25 retrieval
        # ---------------------------------------------------------

        query_tokens = self._tokenize(
            clean_complaint
        )

        bm25_scores = self.bm25.get_scores(
            query_tokens
        )

        bm25_order = np.argsort(
            -bm25_scores
        )

        # ---------------------------------------------------------
        # Hybrid RRF retrieval
        # ---------------------------------------------------------

        hybrid_order = self._rrf(
            dense_order,
            bm25_order,
        )

        # ---------------------------------------------------------
        # Feedback-aware reranking
        # ---------------------------------------------------------

        hybrid_order = (
            self._apply_feedback_reranking(
                hybrid_order,
                clean_complaint,
            )
        )

        # ---------------------------------------------------------
        # Build results
        # ---------------------------------------------------------

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
                    "feedback_score": float(
                        self.feedback_store.get_score(
                            query=clean_complaint,
                            kb_id=article["id"],
                        )
                    ),
                }
            )

        latency_ms = round(
            (
                time.perf_counter()
                - start
            ) * 1000,
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
                    "latency_ms": latency_ms,
                }
            },
        )

        return results


# ================================================================
# FEEDBACK STORE
# ================================================================

class FeedbackStore:
    """
    Lightweight JSONL feedback store.

    Each record looks like:

    {
        "query": "...",
        "kb_id": "KB-1003",
        "feedback": 1
    }

    This intentionally uses JSONL so it is:
    - simple
    - reproducible
    - easy to inspect
    - easy to version
    - independent of the KB
    """

    def __init__(
        self,
        path=DEFAULT_FEEDBACK_PATH,
    ):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.records = self._load()

    def _load(self):
        if not self.path.exists():
            return []

        records = []

        try:
            for line in self.path.read_text(
                encoding="utf-8"
            ).splitlines():

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                    if (
                        "query" in record
                        and "kb_id" in record
                        and "feedback" in record
                    ):
                        records.append(record)

                except json.JSONDecodeError:
                    logger.warning(
                        "invalid_feedback_record"
                    )

        except OSError:
            logger.exception(
                "feedback_store_read_failed"
            )

        return records

    def add_feedback(
        self,
        query: str,
        kb_id: str,
        feedback: int,
    ):
        record = {
            "query": query,
            "kb_id": kb_id,
            "feedback": int(feedback),
        }

        self.records.append(record)

        try:
            with self.path.open(
                "a",
                encoding="utf-8",
            ) as file:

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        except OSError:
            logger.exception(
                "feedback_store_write_failed"
            )

    def get_score(
        self,
        query: str,
        kb_id: str,
    ) -> float:
        """
        Calculate a simple query-aware feedback score.

        Exact normalized query match is used intentionally.
        This prevents unrelated complaints from influencing
        one another.

        Returns a value between -1 and +1.
        """

        normalized_query = (
            self._normalize_query(query)
        )

        relevant = []

        for record in self.records:

            if record.get("kb_id") != kb_id:
                continue

            record_query = self._normalize_query(
                record.get("query", "")
            )

            if record_query == normalized_query:
                relevant.append(
                    int(
                        record.get(
                            "feedback",
                            0,
                        )
                    )
                )

        if not relevant:
            return 0.0

        return float(
            sum(relevant)
            / len(relevant)
        )

    @staticmethod
    def _normalize_query(query):
        return " ".join(
            str(query)
            .lower()
            .strip()
            .split()
        )