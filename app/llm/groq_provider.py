import random
import time
from typing import List

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from app.llm.base import LLMError, LLMResponse, LLMUnavailableError


class GroqProvider:
    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
    ):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")

        self.model = model
        self._client = Groq(
            api_key=api_key,
            timeout=timeout_s,
        )

        # Do not block the evaluation process for very long
        # server-requested rate-limit windows.
        self.max_rate_limit_wait_s = 60.0

    @staticmethod
    def _get_retry_after(error) -> float | None:
        response = getattr(error, "response", None)

        if response is None:
            return None

        headers = getattr(
            response,
            "headers",
            {},
        ) or {}

        for name in (
            "retry-after",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        ):
            value = headers.get(name)

            if value is None:
                continue

            try:
                value = str(value).strip()

                if value.endswith("ms"):
                    return max(
                        0.0,
                        float(value[:-2]) / 1000.0,
                    )

                if value.endswith("s"):
                    return max(
                        0.0,
                        float(value[:-1]),
                    )

                return max(
                    0.0,
                    float(value),
                )

            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        base = min(
            60.0,
            5.0 * (2 ** attempt),
        )

        jitter = random.uniform(
            0.5,
            2.0,
        )

        return base + jitter

    def complete(
        self,
        messages: List[dict],
        *,
        temperature=0.0,
        max_tokens=2000,
        json_mode=False,
    ):
        kwargs = {}

        if json_mode:
            kwargs["response_format"] = {
                "type": "json_object"
            }

        max_attempts = 4

        for attempt in range(max_attempts):
            start = time.perf_counter()

            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                text = (
                    resp.choices[0].message.content
                    or ""
                ).strip()

                if not text:
                    raise LLMError(
                        "Empty response. "
                        "Reasoning models may need "
                        "a larger max_tokens."
                    )

                usage = resp.usage

                return LLMResponse(
                    text=text,
                    model=self.model,
                    latency_ms=round(
                        (
                            time.perf_counter()
                            - start
                        ) * 1000,
                        1,
                    ),
                    prompt_tokens=getattr(
                        usage,
                        "prompt_tokens",
                        None,
                    ),
                    completion_tokens=getattr(
                        usage,
                        "completion_tokens",
                        None,
                    ),
                )

            # --------------------------------------------------
            # RATE LIMIT
            # --------------------------------------------------

            except RateLimitError as e:

                retry_after = self._get_retry_after(e)

                # Server explicitly tells us how long to wait.
                if retry_after is not None:

                    # Do not block the application for a long
                    # rate-limit window. Let the caller resume
                    # later using its checkpoint.
                    if (
                        retry_after
                        > self.max_rate_limit_wait_s
                    ):
                        raise LLMUnavailableError(
                            "Groq rate limit requires "
                            f"waiting {retry_after:.1f}s; "
                            "retry later."
                        ) from e

                    if attempt == max_attempts - 1:
                        raise LLMUnavailableError(
                            "Groq rate limit persisted "
                            f"after {max_attempts} attempts: "
                            f"{e}"
                        ) from e

                    delay = (
                        retry_after
                        + random.uniform(0.5, 2.0)
                    )

                    print(
                        "  Groq rate limit reached. "
                        f"Retrying in {delay:.1f}s "
                        f"(attempt "
                        f"{attempt + 1}/"
                        f"{max_attempts}, "
                        "server reset hint)"
                    )

                    time.sleep(delay)
                    continue

                # No server reset information available.
                if attempt == max_attempts - 1:
                    raise LLMUnavailableError(
                        "Groq rate limit persisted "
                        f"after {max_attempts} attempts: "
                        f"{e}"
                    ) from e

                delay = self._backoff_delay(
                    attempt
                )

                print(
                    "  Groq rate limit reached. "
                    f"Retrying in {delay:.1f}s "
                    f"(attempt "
                    f"{attempt + 1}/"
                    f"{max_attempts}, "
                    "exponential backoff)"
                )

                time.sleep(delay)

            # --------------------------------------------------
            # CONNECTION ERROR
            # --------------------------------------------------

            except APIConnectionError as e:

                if attempt == max_attempts - 1:
                    raise LLMUnavailableError(
                        "Groq connection failed "
                        f"after {max_attempts} attempts: "
                        f"{e}"
                    ) from e

                delay = self._backoff_delay(
                    attempt
                )

                print(
                    "  Groq connection error. "
                    f"Retrying in {delay:.1f}s "
                    f"(attempt "
                    f"{attempt + 1}/"
                    f"{max_attempts})"
                )

                time.sleep(delay)

            # --------------------------------------------------
            # SERVER ERRORS
            # --------------------------------------------------

            except APIStatusError as e:

                if e.status_code >= 500:

                    if attempt == max_attempts - 1:
                        raise LLMUnavailableError(
                            "Groq server error persisted "
                            f"after {max_attempts} attempts: "
                            f"{e}"
                        ) from e

                    delay = self._backoff_delay(
                        attempt
                    )

                    print(
                        f"  Groq server error "
                        f"({e.status_code}). "
                        f"Retrying in {delay:.1f}s "
                        f"(attempt "
                        f"{attempt + 1}/"
                        f"{max_attempts})"
                    )

                    time.sleep(delay)
                    continue

                raise LLMError(
                    str(e)
                ) from e