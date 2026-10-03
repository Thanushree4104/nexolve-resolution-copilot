import logging
import time
import uuid

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import request_id_ctx, setup_logging
from app.llm.factory import get_provider
from app.llm.base import LLMUnavailableError
from app.services.rag import RAGAnswerService
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


# Load components once when the application starts.
retriever = HybridRetriever()

provider = get_provider()

rag_service = RAGAnswerService(
    provider=provider,
    retriever=retriever,
    max_articles=2,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = (
        request.headers.get("X-Request-ID")
        or uuid.uuid4().hex[:12]
    )[:64]

    token = request_id_ctx.set(request_id)
    start = time.perf_counter()

    try:
        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

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
        request_id_ctx.reset(token)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "nexolve-resolution-copilot",
    }


@app.post("/resolve", response_model=ResolveResponse)
def resolve(request: ResolveRequest) -> ResolveResponse:
    start = time.perf_counter()

    try:
        answer = rag_service.answer(request.complaint)

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
            detail="LLM service is temporarily unavailable. Please retry later.",
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