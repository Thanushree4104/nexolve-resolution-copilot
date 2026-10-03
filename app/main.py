import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import (
    request_id_ctx,
    setup_logging,
    trace_id_ctx,
)
from app.llm.base import LLMUnavailableError
from app.llm.factory import get_provider
from app.services.rag_answer import RAGAnswerService
from app.services.retriever import HybridRetriever


setup_logging(settings.log_level)

logger = logging.getLogger("app")


app = FastAPI(
    title="Nexolve Resolution Copilot",
    version="1.0.0",
)


class ResolveRequest(BaseModel):
    complaint: str = Field(
        min_length=5,
        description="Customer complaint",
    )


class ResolveResponse(BaseModel):
    complaint: str
    answer: str


retriever = HybridRetriever()

provider = get_provider()

rag_service = RAGAnswerService(
    provider=provider,
    retriever=retriever,
    max_articles=2,
)


@app.middleware("http")
async def add_request_id(
    request: Request,
    call_next,
):
    request_id = (
        request.headers.get("X-Request-ID")
        or uuid.uuid4().hex[:12]
    )[:64]

    trace_id = uuid.uuid4().hex

    request_token = request_id_ctx.set(
        request_id
    )

    trace_token = trace_id_ctx.set(
        trace_id
    )

    start = time.perf_counter()

    try:
        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            1,
        )

        logger.info(
            "request_completed",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": latency_ms,
                }
            },
        )

        return response

    finally:
        trace_id_ctx.reset(trace_token)
        request_id_ctx.reset(request_token)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "nexolve-resolution-copilot",
    }


@app.post(
    "/resolve",
    response_model=ResolveResponse,
)
def resolve(
    request: ResolveRequest,
) -> ResolveResponse:
    start = time.perf_counter()

    try:
        answer = rag_service.answer(
            request.complaint
        )

    except LLMUnavailableError as e:
        logger.warning(
            "llm_unavailable",
            extra={
                "extra_fields": {
                    "error": str(e),
                }
            },
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "LLM service is temporarily "
                "unavailable. Please retry later."
            ),
        )

    latency_ms = round(
        (time.perf_counter() - start) * 1000,
        1,
    )

    logger.info(
        "resolution_generated",
        extra={
            "extra_fields": {
                "latency_ms": latency_ms,
            }
        },
    )

    return ResolveResponse(
        complaint=request.complaint,
        answer=answer,
    )