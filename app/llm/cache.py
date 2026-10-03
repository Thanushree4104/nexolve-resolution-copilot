import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from app.llm.base import LLMResponse


logger = logging.getLogger("app.llm.cache")


class CachedProvider:
    """Wraps any provider; identical requests are answered from disk."""

    def __init__(
        self,
        inner,
        cache_dir: str = ".cache/llm",
    ):
        self.inner = inner
        self.name = inner.name
        self.model = inner.model

        self.dir = Path(cache_dir)
        self.dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _key(
        self,
        messages,
        temperature,
        max_tokens,
        json_mode,
    ) -> str:
        raw = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            },
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    def complete(
        self,
        messages,
        *,
        temperature=0.0,
        max_tokens=2000,
        json_mode=False,
    ):
        cache_key = self._key(
            messages,
            temperature,
            max_tokens,
            json_mode,
        )

        path = self.dir / f"{cache_key}.json"

        # --------------------------------------------------
        # CACHE HIT
        # --------------------------------------------------

        if path.exists():
            start = time.perf_counter()

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            data["cached"] = True

            latency_ms = round(
                (
                    time.perf_counter()
                    - start
                ) * 1000,
                1,
            )

            data["latency_ms"] = latency_ms

            logger.info(
                "llm_cache_hit",
                extra={
                    "extra_fields": {
                        "provider": self.name,
                        "model": self.model,
                        "cache_key": cache_key,
                        "latency_ms": latency_ms,
                    }
                },
            )

            return LLMResponse(**data)

        # --------------------------------------------------
        # CACHE MISS
        # --------------------------------------------------

        logger.info(
            "llm_cache_miss",
            extra={
                "extra_fields": {
                    "provider": self.name,
                    "model": self.model,
                    "cache_key": cache_key,
                }
            },
        )

        response = self.inner.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

        # --------------------------------------------------
        # STORE RESPONSE
        # --------------------------------------------------

        path.write_text(
            json.dumps(
                asdict(response),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "llm_cache_store",
            extra={
                "extra_fields": {
                    "provider": self.name,
                    "model": self.model,
                    "cache_key": cache_key,
                }
            },
        )

        return response