import pytest

from app.llm.base import LLMResponse, LLMUnavailableError
from app.llm.cache import CachedProvider
from app.llm.mock import MockProvider

MSG = [{"role": "user", "content": "hello"}]


def test_mock_returns_response():
    assert MockProvider(response="hi").complete(MSG).text == "hi"


def test_mock_outage_raises():
    with pytest.raises(LLMUnavailableError):
        MockProvider(fail=True).complete(MSG)


class CountingProvider:
    name = "counting"
    model = "counting-1"

    def __init__(self):
        self.count = 0

    def complete(self, messages, **kwargs):
        self.count += 1
        return LLMResponse(text="answer", model=self.model, latency_ms=500.0)


def test_cache_avoids_second_call(tmp_path):
    inner = CountingProvider()
    cached = CachedProvider(inner, str(tmp_path))
    first = cached.complete(MSG)
    second = cached.complete(MSG)
    assert inner.count == 1
    assert first.cached is False
    assert second.cached is True
    assert second.latency_ms < 500


def test_cache_miss_on_different_prompt(tmp_path):
    inner = CountingProvider()
    cached = CachedProvider(inner, str(tmp_path))
    cached.complete(MSG)
    cached.complete([{"role": "user", "content": "different"}])
    assert inner.count == 2