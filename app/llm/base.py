from dataclasses import dataclass
from typing import List, Optional, Protocol


class LLMError(Exception):
    """The LLM call failed (bad request, empty output, etc.)."""


class LLMUnavailableError(LLMError):
    """The LLM is unreachable, rate-limited or timing out. Triggers fallbacks."""


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool = False
    latency_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self,
        messages: List[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> LLMResponse: ...