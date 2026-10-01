import time
from typing import List

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from app.llm.base import LLMError, LLMResponse, LLMUnavailableError


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str, model: str, timeout_s: float = 30.0):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self.model = model
        self._client = Groq(api_key=api_key, timeout=timeout_s)

    def complete(self, messages: List[dict], *, temperature=0.0, max_tokens=2000, json_mode=False):
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except (APIConnectionError, RateLimitError) as e:
            raise LLMUnavailableError(str(e)) from e
        except APIStatusError as e:
            if e.status_code >= 500:
                raise LLMUnavailableError(str(e)) from e
            raise LLMError(str(e)) from e

        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise LLMError("Empty response. Reasoning models may need a larger max_tokens.")
        usage = resp.usage
        return LLMResponse(
            text=text,
            model=self.model,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )