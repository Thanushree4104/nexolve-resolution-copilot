import logging
import time
import uuid

from fastapi import FastAPI, Request

from app.core.config import settings
from app.core.logging import request_id_ctx, setup_logging

setup_logging(settings.log_level)
logger = logging.getLogger("app")

app = FastAPI(title="Nexolve Resolution Copilot")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = (request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12])[:64]
    token = request_id_ctx.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
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
    return {"status": "ok"}