from typing import List

from app.llm.base import LLMResponse, LLMUnavailableError


class MockProvider:
    """Deterministic fake LLM for tests and CI. Set fail=True to simulate an outage."""

    name = "mock"
    model = "mock"

    def __init__(self, response: str = '{"mock": true}', fail: bool = False):
        self.response = response
        self.fail = fail
        self.calls: List[list] = []  # records every prompt it received

    def complete(self, messages, *, temperature=0.0, max_tokens=2000, json_mode=False):
        self.calls.append(messages)
        if self.fail:
            raise LLMUnavailableError("mock outage")
        return LLMResponse(text=self.response, model=self.model)