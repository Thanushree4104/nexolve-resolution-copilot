import json
import logging
import math
import re
from pathlib import Path
from typing import Any


logger = logging.getLogger("app.feedback")


# ============================================================
# FEEDBACK STORE
# ============================================================


class FeedbackStore:

    def __init__(
        self,
        path: str = "data/feedback/retrieval_feedback.jsonl",
    ):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # QUERY NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_query(
        query: str,
    ) -> str:

        query = str(
            query or ""
        ).strip().lower()

        query = re.sub(
            r"\s+",
            " ",
            query,
        )

        return query

    # ========================================================
    # QUERY TOKENIZATION
    # ========================================================

    @staticmethod
    def tokenize_query(
        query: str,
    ) -> set[str]:

        normalized = (
            FeedbackStore.normalize_query(
                query
            )
        )

        # Keep meaningful alphanumeric tokens.
        tokens = re.findall(
            r"[a-z0-9]+",
            normalized,
        )

        return set(tokens)

    # ========================================================
    # QUERY SEMANTIC SIMILARITY
    # ========================================================

    @staticmethod
    def query_similarity(
        query_a: str,
        query_b: str,
    ) -> float:

        tokens_a = (
            FeedbackStore.tokenize_query(
                query_a
            )
        )

        tokens_b = (
            FeedbackStore.tokenize_query(
                query_b
            )
        )

        if not tokens_a or not tokens_b:
            return 0.0

        # ----------------------------------------------------
        # Jaccard similarity.
        #
        # similarity =
        # intersection / union
        # ----------------------------------------------------

        intersection = len(
            tokens_a & tokens_b
        )

        union = len(
            tokens_a | tokens_b
        )

        if union == 0:
            return 0.0

        jaccard = (
            intersection
            / union
        )

        # ----------------------------------------------------
        # Token overlap coefficient.
        #
        # This helps when one query is shorter than another.
        #
        # Example:
        #
        # "internet disconnecting"
        #
        # vs
        #
        # "internet keeps disconnecting"
        #
        # Jaccard can be conservative, while overlap
        # coefficient recognizes the strong shared intent.
        # ----------------------------------------------------

        minimum_size = min(
            len(tokens_a),
            len(tokens_b),
        )

        if minimum_size == 0:
            return 0.0

        overlap = (
            intersection
            / minimum_size
        )

        # ----------------------------------------------------
        # Combine the two measures.
        #
        # Jaccard prevents overly aggressive matching.
        # Overlap helps short queries.
        # ----------------------------------------------------

        similarity = (
            0.6 * jaccard
            + 0.4 * overlap
        )

        return float(
            max(
                0.0,
                min(
                    similarity,
                    1.0,
                ),
            )
        )

    # ========================================================
    # APPEND FEEDBACK
    # ========================================================

    def add_feedback(
        self,
        query: str,
        kb_id: str,
        feedback: int,
    ) -> None:

        normalized_query = (
            self.normalize_query(
                query
            )
        )

        normalized_kb_id = str(
            kb_id or ""
        ).strip().upper()

        feedback_value = int(
            feedback
        )

        if not normalized_query:
            raise ValueError(
                "Feedback query cannot be empty."
            )

        if not normalized_kb_id:
            raise ValueError(
                "Feedback KB ID cannot be empty."
            )

        if feedback_value not in (
            -1,
            1,
        ):
            raise ValueError(
                "Feedback must be either -1 or 1."
            )

        record = {
            "query": normalized_query,
            "kb_id": normalized_kb_id,
            "feedback": feedback_value,
        }

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

    # ========================================================
    # READ ALL FEEDBACK
    # ========================================================

    def _read_feedback(
        self,
    ) -> list[dict[str, Any]]:

        if not self.path.exists():
            return []

        records = []

        try:

            with self.path.open(
                "r",
                encoding="utf-8",
            ) as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue

                    try:

                        record = json.loads(
                            line
                        )

                    except json.JSONDecodeError:

                        logger.warning(
                            "invalid_feedback_record",
                            extra={
                                "extra_fields": {
                                    "line": line[:300],
                                }
                            },
                        )

                        continue

                    if not isinstance(
                        record,
                        dict,
                    ):
                        continue

                    records.append(
                        record
                    )

        except OSError as exc:

            logger.warning(
                "feedback_store_read_failed",
                extra={
                    "extra_fields": {
                        "error": str(exc),
                        "path": str(self.path),
                    }
                },
            )

        return records

    # ========================================================
    # QUERY-SPECIFIC FEEDBACK
    # ========================================================

    def get_query_feedback(
        self,
        query: str,
    ) -> dict[str, float]:

        normalized_query = (
            self.normalize_query(
                query
            )
        )

        scores: dict[str, float] = {}

        for record in self._read_feedback():

            record_query = (
                self.normalize_query(
                    record.get(
                        "query",
                        "",
                    )
                )
            )

            if record_query != normalized_query:
                continue

            kb_id = str(
                record.get(
                    "kb_id",
                    "",
                )
            ).strip().upper()

            if not kb_id:
                continue

            try:

                feedback = float(
                    record.get(
                        "feedback",
                        0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            scores[kb_id] = (
                scores.get(
                    kb_id,
                    0.0,
                )
                + feedback
            )

        return scores

    # ========================================================
    # SEMANTIC QUERY FEEDBACK
    # ========================================================

    def get_semantic_feedback(
        self,
        query: str,
        similarity_threshold: float = 0.45,
    ) -> dict[str, dict[str, float]]:

        normalized_query = (
            self.normalize_query(
                query
            )
        )

        results: dict[
            str,
            dict[str, float],
        ] = {}

        for record in self._read_feedback():

            record_query = (
                self.normalize_query(
                    record.get(
                        "query",
                        "",
                    )
                )
            )

            if not record_query:
                continue

            # Exact matches are handled separately.
            if record_query == normalized_query:
                continue

            similarity = (
                self.query_similarity(
                    normalized_query,
                    record_query,
                )
            )

            if similarity < similarity_threshold:
                continue

            kb_id = str(
                record.get(
                    "kb_id",
                    "",
                )
            ).strip().upper()

            if not kb_id:
                continue

            try:

                feedback = float(
                    record.get(
                        "feedback",
                        0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            # ------------------------------------------------
            # Keep the strongest matching historical query
            # for each KB article.
            #
            # This prevents several weakly similar queries
            # from overwhelming one strong match.
            # ------------------------------------------------

            existing = results.get(
                kb_id
            )

            if (
                existing is None
                or similarity
                > existing[
                    "similarity"
                ]
            ):

                results[kb_id] = {
                    "feedback": feedback,
                    "similarity": similarity,
                    "source_query": record_query,
                }

        return results


# ============================================================
# FEEDBACK RERANKER
# ============================================================


class FeedbackReranker:

    def __init__(
        self,
        feedback_store: FeedbackStore | None = None,
        feedback_weight: float = 0.15,
        semantic_feedback_weight: float = 0.50,
        semantic_similarity_threshold: float = 0.45,
    ):

        self.feedback_store = (
            feedback_store
            or FeedbackStore()
        )

        # ----------------------------------------------------
        # Main exact-query feedback weight.
        # ----------------------------------------------------

        configured_weight = float(
            feedback_weight
        )

        self.feedback_weight = min(
            max(
                configured_weight,
                0.0,
            ),
            0.25,
        )

        if (
            configured_weight
            != self.feedback_weight
        ):

            logger.warning(
                "feedback_weight_clipped",
                extra={
                    "extra_fields": {
                        "configured_weight": (
                            configured_weight
                        ),
                        "effective_weight": (
                            self.feedback_weight
                        ),
                        "minimum": 0.0,
                        "maximum": 0.25,
                    }
                },
            )

        # ----------------------------------------------------
        # Semantic feedback is deliberately weaker.
        #
        # Example:
        #
        # exact feedback:
        #
        #     0.15
        #
        # semantic feedback:
        #
        #     0.15 * 0.50 * similarity
        #
        # This prevents approximate matches from dominating
        # the original retrieval score.
        # ----------------------------------------------------

        configured_semantic_weight = float(
            semantic_feedback_weight
        )

        self.semantic_feedback_weight = min(
            max(
                configured_semantic_weight,
                0.0,
            ),
            1.0,
        )

        if (
            configured_semantic_weight
            != self.semantic_feedback_weight
        ):

            logger.warning(
                "semantic_feedback_weight_clipped",
                extra={
                    "extra_fields": {
                        "configured_weight": (
                            configured_semantic_weight
                        ),
                        "effective_weight": (
                            self.semantic_feedback_weight
                        ),
                        "minimum": 0.0,
                        "maximum": 1.0,
                    }
                },
            )

        # ----------------------------------------------------
        # Minimum similarity required before historical
        # feedback can influence the current query.
        # ----------------------------------------------------

        configured_threshold = float(
            semantic_similarity_threshold
        )

        self.semantic_similarity_threshold = min(
            max(
                configured_threshold,
                0.0,
            ),
            1.0,
        )

        if (
            configured_threshold
            != self.semantic_similarity_threshold
        ):

            logger.warning(
                "semantic_feedback_threshold_clipped",
                extra={
                    "extra_fields": {
                        "configured_threshold": (
                            configured_threshold
                        ),
                        "effective_threshold": (
                            self.semantic_similarity_threshold
                        ),
                        "minimum": 0.0,
                        "maximum": 1.0,
                    }
                },
            )

    # ============================================================
    # RERANK
    # ============================================================

    def rerank(
        self,
        query: str,
        articles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        if not articles:
            return []

        normalized_query = (
            FeedbackStore.normalize_query(
                query
            )
        )

        # ========================================================
        # EXACT QUERY FEEDBACK
        # ========================================================

        exact_feedback = (
            self.feedback_store.get_query_feedback(
                query
            )
        )

        # ========================================================
        # SEMANTIC QUERY FEEDBACK
        # ========================================================

        semantic_feedback = (
            self.feedback_store.get_semantic_feedback(
                query,
                similarity_threshold=(
                    self.semantic_similarity_threshold
                ),
            )
        )

        # ========================================================
        # CANDIDATE KB IDS
        # ========================================================

        candidate_ids = set()

        for article in articles:

            kb_id = str(
                article.get(
                    "id",
                    "",
                )
            ).strip().upper()

            if kb_id:
                candidate_ids.add(
                    kb_id
                )

        # ========================================================
        # COLLECT FEEDBACK FOR CANDIDATES
        # ========================================================

        feedback_scores: dict[
            str,
            float,
        ] = {}

        feedback_sources: dict[
            str,
            dict[str, Any],
        ] = {}

        for kb_id in candidate_ids:

            # ----------------------------------------------------
            # Exact feedback has priority.
            # ----------------------------------------------------

            if kb_id in exact_feedback:

                score = float(
                    exact_feedback[
                        kb_id
                    ]
                )

                feedback_scores[
                    kb_id
                ] = score

                feedback_sources[
                    kb_id
                ] = {
                    "source": "exact",
                    "similarity": 1.0,
                    "feedback": score,
                    "source_query": (
                        normalized_query
                    ),
                }

                continue

            # ----------------------------------------------------
            # Semantic feedback is used only when no exact
            # feedback exists for this candidate.
            # ----------------------------------------------------

            semantic = (
                semantic_feedback.get(
                    kb_id
                )
            )

            if semantic is not None:

                score = float(
                    semantic.get(
                        "feedback",
                        0.0,
                    )
                )

                similarity = float(
                    semantic.get(
                        "similarity",
                        0.0,
                    )
                )

                feedback_scores[
                    kb_id
                ] = score

                feedback_sources[
                    kb_id
                ] = {
                    "source": "semantic",
                    "similarity": similarity,
                    "feedback": score,
                    "source_query": semantic.get(
                        "source_query",
                        "",
                    ),
                }

            else:

                feedback_scores[
                    kb_id
                ] = 0.0

                feedback_sources[
                    kb_id
                ] = {
                    "source": "none",
                    "similarity": 0.0,
                    "feedback": 0.0,
                    "source_query": None,
                }

        # ========================================================
        # NORMALIZE FEEDBACK
        # ========================================================

        max_feedback = max(
            (
                abs(score)
                for score in feedback_scores.values()
            ),
            default=0.0,
        )

        reranked = []

        for article in articles:

            kb_id = str(
                article.get(
                    "id",
                    "",
                )
            ).strip().upper()

            feedback_score = float(
                feedback_scores.get(
                    kb_id,
                    0.0,
                )
            )

            source_info = (
                feedback_sources.get(
                    kb_id,
                    {
                        "source": "none",
                        "similarity": 0.0,
                        "feedback": 0.0,
                        "source_query": None,
                    },
                )
            )

            feedback_source = (
                source_info.get(
                    "source",
                    "none",
                )
            )

            similarity = float(
                source_info.get(
                    "similarity",
                    0.0,
                )
            )

            # ----------------------------------------------------
            # Normalize feedback.
            # ----------------------------------------------------

            if max_feedback > 0:

                normalized_feedback = (
                    feedback_score
                    / max_feedback
                )

            else:

                normalized_feedback = 0.0

            # ----------------------------------------------------
            # Existing hybrid retrieval score.
            # ----------------------------------------------------

            try:

                base_score = float(
                    article.get(
                        "hybrid_score",
                        0.0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                base_score = 0.0

            # ====================================================
            # FEEDBACK CONTRIBUTION
            # ====================================================

            if feedback_source == "exact":

                feedback_contribution = (
                    self.feedback_weight
                    * normalized_feedback
                )

            elif feedback_source == "semantic":

                # ------------------------------------------------
                # Semantic feedback is discounted by:
                #
                # 1. semantic_feedback_weight
                # 2. query similarity
                #
                # Example:
                #
                # feedback_weight = 0.15
                # semantic_weight = 0.50
                # similarity = 0.70
                #
                # contribution =
                #
                # 0.15 * 0.50 * 0.70
                #
                # = 0.0525
                # ------------------------------------------------

                feedback_contribution = (
                    self.feedback_weight
                    * self.semantic_feedback_weight
                    * similarity
                    * normalized_feedback
                )

            else:

                feedback_contribution = 0.0

            # ====================================================
            # FINAL SCORE
            # ====================================================

            rerank_score = (
                base_score
                + feedback_contribution
            )

            updated = dict(
                article
            )

            updated[
                "feedback_score"
            ] = feedback_score

            updated[
                "feedback_normalized"
            ] = normalized_feedback

            updated[
                "feedback_contribution"
            ] = feedback_contribution

            updated[
                "feedback_source"
            ] = feedback_source

            updated[
                "feedback_query_similarity"
            ] = similarity

            updated[
                "feedback_source_query"
            ] = source_info.get(
                "source_query"
            )

            updated[
                "rerank_score"
            ] = rerank_score

            reranked.append(
                updated
            )

        # ========================================================
        # SORT
        # ========================================================

        reranked.sort(
            key=lambda item: float(
                item.get(
                    "rerank_score",
                    0.0,
                )
            ),
            reverse=True,
        )

        # ========================================================
        # FINAL RANKS
        # ========================================================

        for rank, article in enumerate(
            reranked,
            start=1,
        ):

            article["rank"] = rank

        # ========================================================
        # STRUCTURED OBSERVABILITY
        # ========================================================

        candidate_feedback = {}

        for article in reranked:

            kb_id = article.get(
                "id",
                "",
            )

            candidate_feedback[
                kb_id
            ] = {
                "feedback_score": article.get(
                    "feedback_score",
                    0.0,
                ),
                "feedback_normalized": article.get(
                    "feedback_normalized",
                    0.0,
                ),
                "feedback_contribution": article.get(
                    "feedback_contribution",
                    0.0,
                ),
                "feedback_source": article.get(
                    "feedback_source",
                    "none",
                ),
                "feedback_query_similarity": article.get(
                    "feedback_query_similarity",
                    0.0,
                ),
                "feedback_source_query": article.get(
                    "feedback_source_query",
                ),
                "hybrid_score": article.get(
                    "hybrid_score",
                    0.0,
                ),
                "rerank_score": article.get(
                    "rerank_score",
                    0.0,
                ),
            }

        final_order = [
            article.get(
                "id",
                "",
            )
            for article in reranked
        ]

        # ========================================================
        # LOG
        # ========================================================

        logger.info(
            "feedback_reranking_completed",
            extra={
                "extra_fields": {
                    "query": normalized_query,
                    "feedback_weight": (
                        self.feedback_weight
                    ),
                    "semantic_feedback_weight": (
                        self.semantic_feedback_weight
                    ),
                    "semantic_similarity_threshold": (
                        self.semantic_similarity_threshold
                    ),
                    "exact_query_feedback": (
                        exact_feedback
                    ),
                    "semantic_query_feedback": (
                        semantic_feedback
                    ),
                    "candidate_feedback": (
                        candidate_feedback
                    ),
                    "final_order": (
                        final_order
                    ),
                }
            },
        )

        return reranked