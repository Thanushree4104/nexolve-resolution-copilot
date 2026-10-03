import logging
import random
import time
from typing import List

from groq import (
    APIConnectionError,
    APIStatusError,
    Groq,
    RateLimitError,
)

from app.llm.base import (
    LLMError,
    LLMResponse,
    LLMUnavailableError,
)


logger = logging.getLogger("app.llm.groq")


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

        self.max_rate_limit_wait_s = 60.0
        self.max_attempts = 4

    @staticmethod
    def _get_retry_after(error) -> float | None:
        response = getattr(
            error,
            "response",
            None,
        )

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
        temperature: float = 0.0,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> LLMResponse:

        kwargs = {}

        if json_mode:
            kwargs["response_format"] = {
                "type": "json_object"
            }

        for attempt in range(self.max_attempts):
            start = time.perf_counter()

            try:
                logger.info(
                    "llm_request_started",
                    extra={
                        "extra_fields": {
                            "provider": self.name,
                            "model": self.model,
                            "attempt": attempt + 1,
                            "max_attempts": self.max_attempts,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "json_mode": json_mode,
                        }
                    },
                )

                response = (
                    self._client
                    .chat
                    .completions
                    .create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                )

                latency_ms = round(
                    (
                        time.perf_counter()
                        - start
                    ) * 1000,
                    1,
                )

                text = (
                    response
                    .choices[0]
                    .message
                    .content
                    or ""
                ).strip()

                if not text:
                    logger.error(
                        "llm_empty_response",
                        extra={
                            "extra_fields": {
                                "provider": self.name,
                                "model": self.model,
                                "attempt": attempt + 1,
                                "latency_ms": latency_ms,
                            }
                        },
                    )

                    raise LLMError(
                        "Empty response from LLM"
                    )

                usage = response.usage

                prompt_tokens = getattr(
                    usage,
                    "prompt_tokens",
                    None,
                )

                completion_tokens = getattr(
                    usage,
                    "completion_tokens",
                    None,
                )

                logger.info(
                    "llm_request_completed",
                    extra={
                        "extra_fields": {
                            "provider": self.name,
                            "model": self.model,
                            "attempt": attempt + 1,
                            "retry_count": attempt,
                            "latency_ms": latency_ms,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        }
                    },
                )

                return LLMResponse(
                    text=text,
                    model=self.model,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            except RateLimitError as error:

                retry_after = self._get_retry_after(
                    error
                )

                if retry_after is not None:

                    if (
                        retry_after
                        > self.max_rate_limit_wait_s
                    ):
                        logger.warning(
                            "llm_rate_limit_unavailable",
                            extra={
                                "extra_fields": {
                                    "provider": self.name,
                                    "model": self.model,
                                    "attempt": attempt + 1,
                                    "retry_after_s": retry_after,
                                    "reason": "retry_window_too_long",
                                }
                            },
                        )

                        raise LLMUnavailableError(
                            "Groq rate limit requires "
                            f"waiting {retry_after:.1f}s; "
                            "retry later."
                        ) from error

                    if (
                        attempt
                        == self.max_attempts - 1
                    ):
                        logger.error(
                            "llm_rate_limit_exhausted",
                            extra={
                                "extra_fields": {
                                    "provider": self.name,
                                    "model": self.model,
                                    "attempt": attempt + 1,
                                    "retry_count": attempt,
                                    "reason": "rate_limit",
                                }
                            },
                        )

                        raise LLMUnavailableError(
                            "Groq rate limit persisted "
                            f"after {self.max_attempts} attempts."
                        ) from error

                    delay = (
                        retry_after
                        + random.uniform(
                            0.5,
                            2.0,
                        )
                    )

                    logger.warning(
                        "llm_retry",
                        extra={
                            "extra_fields": {
                                "provider": self.name,
                                "model": self.model,
                                "attempt": attempt + 1,
                                "retry_count": attempt + 1,
                                "reason": "rate_limit",
                                "retry_after_s": retry_after,
                                "delay_s": round(
                                    delay,
                                    2,
                                ),
                            }
                        },
                    )

                    time.sleep(delay)
                    continue

                if (
                    attempt
                    == self.max_attempts - 1
                ):
                    logger.error(
                        "llm_rate_limit_exhausted",
                        extra={
                            "extra_fields": {
                                "provider": self.name,
                                "model": self.model,
                                "attempt": attempt + 1,
                                "retry_count": attempt,
                                "reason": "rate_limit",
                            }
                        },
                    )

                    raise LLMUnavailableError(
                        "Groq rate limit persisted "
                        f"after {self.max_attempts} attempts."
                    ) from error

                delay = self._backoff_delay(
                    attempt
                )

                logger.warning(
                    "llm_retry",
                    extra={
                        "extra_fields": {
                            "provider": self.name,
                            "model": self.model,
                            "attempt": attempt + 1,
                            "retry_count": attempt + 1,
                            "reason": "rate_limit",
                            "delay_s": round(
                                delay,
                                2,
                            ),
                        }
                    },
                )

                time.sleep(delay)

            except APIConnectionError as error:

                if (
                    attempt
                    == self.max_attempts - 1
                ):
                    logger.error(
                        "llm_connection_failed",
                        extra={
                            "extra_fields": {
                                "provider": self.name,
                                "model": self.model,
                                "attempt": attempt + 1,
                                "retry_count": attempt,
                            }
                        },
                    )

                    raise LLMUnavailableError(
                        "Groq connection failed "
                        f"after {self.max_attempts} attempts."
                    ) from error

                delay = self._backoff_delay(
                    attempt
                )

                logger.warning(
                    "llm_retry",
                    extra={
                        "extra_fields": {
                            "provider": self.name,
                            "model": self.model,
                            "attempt": attempt + 1,
                            "retry_count": attempt + 1,
                            "reason": "connection_error",
                            "delay_s": round(
                                delay,
                                2,
                            ),
                        }
                    },
                )

                time.sleep(delay)

            except APIStatusError as error:

                if error.status_code >= 500:

                    if (
                        attempt
                        == self.max_attempts - 1
                    ):
                        logger.error(
                            "llm_server_error",
                            extra={
                                "extra_fields": {
                                    "provider": self.name,
                                    "model": self.model,
                                    "attempt": attempt + 1,
                                    "retry_count": attempt,
                                    "status_code": error.status_code,
                                }
                            },
                        )

                        raise LLMUnavailableError(
                            "Groq server error persisted "
                            f"after {self.max_attempts} attempts."
                        ) from error

                    delay = self._backoff_delay(
                        attempt
                    )

                    logger.warning(
                        "llm_retry",
                        extra={
                            "extra_fields": {
                                "provider": self.name,
                                "model": self.model,
                                "attempt": attempt + 1,
                                "retry_count": attempt + 1,
                                "reason": "server_error",
                                "status_code": error.status_code,
                                "delay_s": round(
                                    delay,
                                    2,
                                ),
                            }
                        },
                    )

                    time.sleep(delay)
                    continue

                logger.error(
                    "llm_request_failed",
                    extra={
                        "extra_fields": {
                            "provider": self.name,
                            "model": self.model,
                            "attempt": attempt + 1,
                            "status_code": error.status_code,
                            "reason": "client_error",
                        }
                    },
                )

                raise LLMError(
                    str(error)
                ) from error