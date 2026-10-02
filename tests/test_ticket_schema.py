import json

import pytest

from app.schemas.ticket import parse_batch

ONE = {"complaint": "Internet drops every night after dinner time.", "resolution_notes": "fixed", "steps_taken": []}


def test_parses_expected_count():
    text = json.dumps({"tickets": [ONE, ONE]})
    assert len(parse_batch(text, 2)) == 2


def test_rejects_wrong_count():
    with pytest.raises(ValueError):
        parse_batch(json.dumps({"tickets": [ONE]}), 2)


def test_parses_inside_code_fences():
    text = "```json\n" + json.dumps({"tickets": [ONE]}) + "\n```"
    assert len(parse_batch(text, 1)) == 1


def test_rejects_too_short_complaint():
    bad = {**ONE, "complaint": "slow"}
    with pytest.raises(ValueError):
        parse_batch(json.dumps({"tickets": [bad]}), 1)