import json
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class DiagnosticQuestion(BaseModel):
    q: str
    if_yes: str
    if_no: str


class Step(BaseModel):
    n: int
    text: str


class KBBody(BaseModel):
    """What the LLM writes."""

    title: str
    symptoms: List[str] = Field(min_length=3)
    diagnostic_questions: List[DiagnosticQuestion] = Field(min_length=1)
    steps: List[Step] = Field(min_length=4)
    escalate_when: List[str] = Field(min_length=1)
    notes: str = ""


class KBArticle(KBBody):
    """The full record: LLM text plus fields assigned by code."""

    model_config = ConfigDict(extra="forbid")

    id: str
    class_id: str
    product: str
    root_cause: str
    version: int = 1
    last_reviewed: str
    status: str = "active"
    synthetic: bool = True


_TRANSLATE = str.maketrans(
    {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u00a0": " ", "\u202f": " ",
    }
)


def _clean(obj):
    """Replace look-alike characters with plain ASCII, recursively."""
    if isinstance(obj, str):
        return obj.translate(_TRANSLATE)
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    return obj


def parse_body(text: str) -> KBBody:
    """Pull the JSON object out of an LLM reply (it may be wrapped in code fences) and validate it."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in LLM output")
    return KBBody(**_clean(json.loads(text[start : end + 1])))