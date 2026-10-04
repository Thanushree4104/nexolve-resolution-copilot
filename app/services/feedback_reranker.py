from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FeedbackStore:
    def __init__(self, path: str = "data/feedback/retrieval_feedback.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_feedback(
        self,
        query: str,
        kb_id: str,
        feedback: int,
    ) -> None:
        if feedback not in (-1, 1):
            raise ValueError("feedback must be -1 or 1")

        record = {
            "query": query,
            "kb_id": kb_id,
            "feedback": feedback,
        }

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        records = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return records

    def get_kb_score(self, kb_id: str) -> float:
        records = self.load()

        score = 0.0

        for record in records:
            if record.get("kb_id") == kb_id:
                score += float(record.get("feedback", 0))

        return score


class FeedbackReranker:
    def __init__(
        self,
        feedback_store: FeedbackStore | None = None,
        feedback_weight: float = 0.10,
    ):
        self.feedback_store = feedback_store or FeedbackStore()
        self.feedback_weight = feedback_weight

    def rerank(
        self,
        query: str,
        articles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not articles:
            return []

        max_feedback = max(
            (
                abs(
                    self.feedback_store.get_kb_score(
                        article.get("id", "")
                    )
                )
                for article in articles
            ),
            default=0.0,
        )

        reranked = []

        for article in articles:
            kb_id = article.get("id", "")

            feedback_score = self.feedback_store.get_kb_score(
                kb_id
            )

            if max_feedback > 0:
                normalized_feedback = (
                    feedback_score / max_feedback
                )
            else:
                normalized_feedback = 0.0

            base_score = self._base_score(article)

            final_score = (
                base_score
                + self.feedback_weight * normalized_feedback
            )

            updated = dict(article)

            updated["feedback_score"] = feedback_score
            updated["feedback_normalized"] = normalized_feedback
            updated["rerank_score"] = final_score

            reranked.append(updated)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        for rank, article in enumerate(
            reranked,
            start=1,
        ):
            article["rank"] = rank

        return reranked

    @staticmethod
    def _base_score(article: dict[str, Any]) -> float:
        """
        Use the existing retrieval scores.

        RRF score is preferred when available.
        Otherwise fall back to dense + BM25.
        """

        if article.get("rrf_score") is not None:
            return float(article["rrf_score"])

        dense = float(
            article.get("dense_score", 0.0)
        )

        bm25 = float(
            article.get("bm25_score", 0.0)
        )

        return (
            0.5 * dense
            + 0.5 * bm25
        )