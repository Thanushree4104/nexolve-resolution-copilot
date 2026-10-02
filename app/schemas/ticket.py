import json
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.kb import _clean  # same look-alike character cleanup as the KB


class TicketText(BaseModel):
    """What the LLM writes."""

    complaint: str = Field(min_length=15)
    resolution_notes: str = Field(min_length=2)
    steps_taken: List[str] = Field(default_factory=list)


class Ticket(TicketText):
    """The full record: LLM text plus fields assigned by code."""

    model_config = ConfigDict(extra="forbid")

    id: str
    class_id: str
    product: str
    root_cause: str
    kb_id: str
    language: str
    style: str
    persona: str
    resolution_quality: str
    steps_already_tried: List[str]
    reopened: bool
    first_contact_resolution: bool
    created_at: str
    region: str
    firmware: Optional[str] = None
    synthetic: bool = True


def parse_batch(text: str, expected: int) -> List[TicketText]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in LLM output")
    data = _clean(json.loads(text[start : end + 1]))
    items = data.get("tickets") if isinstance(data, dict) else None
    if not isinstance(items, list) or len(items) != expected:
        raise ValueError(f"expected {expected} tickets, got {len(items) if items else 0}")
    return [TicketText(**item) for item in items]