from app.core.pii import redact


class RedactingProvider:
    """Wraps any provider and redacts PII from every message before it goes out."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.model = inner.model

    def complete(self, messages, **kwargs):
        clean = [{**m, "content": redact(m["content"]).text} for m in messages]
        return self.inner.complete(clean, **kwargs)