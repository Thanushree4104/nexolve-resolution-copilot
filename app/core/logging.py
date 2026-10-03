import json
import logging
import sys
from contextvars import ContextVar


request_id_ctx: ContextVar[str] = ContextVar(
    "request_id",
    default="-",
)


trace_id_ctx: ContextVar[str] = ContextVar(
    "trace_id",
    default="-",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(
                record,
                "%Y-%m-%dT%H:%M:%S",
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
            "trace_id": trace_id_ctx.get(),
        }

        extra = getattr(
            record,
            "extra_fields",
            None,
        )

        if extra:
            payload.update(extra)

        return json.dumps(payload)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()

    root.handlers = [handler]
    root.setLevel(level)