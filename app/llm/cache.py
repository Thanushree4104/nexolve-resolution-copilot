import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from app.llm.base import LLMResponse


class CachedProvider:
    """Wraps any provider; identical requests are answered from disk."""

    def __init__(self, inner, cache_dir: str = ".cache/llm"):
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, messages, temperature, max_tokens, json_mode) -> str:
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
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def complete(
        self,
        messages,
        *,
        temperature=0.0,
        max_tokens=2000,
        json_mode=False,
    ):
        path = self.dir / (
            f"{self._key(messages, temperature, max_tokens, json_mode)}.json"
        )

        # Cache hit
        if path.exists():
            start = time.perf_counter()

            data = json.loads(path.read_text(encoding="utf-8"))

            data["cached"] = True
            data["latency_ms"] = round(
                (time.perf_counter() - start) * 1000,
                1,
            )

            return LLMResponse(**data)

        # Cache miss - call the actual LLM provider
        resp = self.inner.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

        # Store response in cache
        path.write_text(
            json.dumps(asdict(resp), ensure_ascii=False),
            encoding="utf-8",
        )

        return resp