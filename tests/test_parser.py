import json

import pytest

from app.core.taxonomy import load_taxonomy
from app.llm.base import LLMError, LLMResponse
from app.llm.mock import MockProvider
from app.llm.redacting import RedactingProvider
from app.services.parser import ComplaintParser, build_prompt

GOOD = {
    "category": "connectivity.intermittent",
    "category_confidence": 0.8,
    "unknown_label": None,
    "product": "Nimbus Fibre",
    "severity": "High",
    "sentiment": "frustrated",
    "symptoms": ["drops every evening around 8"],
    "steps_already_tried": ["restarted router twice"],
    "customer_impact": "works from home",
}


class SeqProvider:
    name = "seq"
    model = "seq"

    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages, **kwargs):
        return LLMResponse(text=self.replies.pop(0), model="seq")


def make(reply):
    return ComplaintParser(MockProvider(response=json.dumps(reply)), load_taxonomy())


def test_parses_good_output_and_lowercases_severity():
    p = make(GOOD).parse("Internet drops every evening.")
    assert p.category == "connectivity.intermittent"
    assert p.severity == "high"
    assert p.steps_already_tried == ["restarted router twice"]


def test_unrecognized_category_becomes_unknown():
    p = make({**GOOD, "category": "tv.streaming"}).parse("My TV box freezes.")
    assert p.category == "unknown"
    assert p.unknown_label == "tv.streaming"


def test_prompt_lists_only_active_classes():
    prompt = build_prompt(load_taxonomy(), "x")
    assert "billing.dispute" in prompt
    assert "tv.streaming" not in prompt


def test_activating_a_class_changes_the_prompt_without_code_changes():
    tax = load_taxonomy().model_copy(deep=True)
    tax.get("tv.streaming").status = "active"
    assert "tv.streaming" in build_prompt(tax, "x")


def test_retry_recovers_from_bad_json():
    provider = SeqProvider(["not json at all", json.dumps(GOOD)])
    p = ComplaintParser(provider, load_taxonomy()).parse("Internet drops.")
    assert p.category == "connectivity.intermittent"


def test_two_bad_outputs_raise_llm_error():
    provider = SeqProvider(["nope", "still nope"])
    with pytest.raises(LLMError):
        ComplaintParser(provider, load_taxonomy()).parse("Internet drops.")


def test_parser_never_sends_pii_to_the_llm():
    mock = MockProvider(response=json.dumps(GOOD))
    parser = ComplaintParser(RedactingProvider(mock), load_taxonomy())
    parser.parse("Call me on +91 98765 43210 or mail ravi@example.com, internet drops.")
    sent = str(mock.calls)
    assert "98765" not in sent
    assert "ravi@example.com" not in sent