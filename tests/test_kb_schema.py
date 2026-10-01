import json

import pytest
from pydantic import ValidationError

from app.schemas.kb import parse_body

GOOD = {
    "title": "Evening drops on a busy node",
    "symptoms": ["slow at night", "video freezes", "calls cut out"],
    "diagnostic_questions": [{"q": "Does it happen at the same time daily?", "if_yes": "a", "if_no": "b"}],
    "steps": [{"n": i, "text": f"step {i}"} for i in range(1, 6)],
    "escalate_when": ["more than 5 neighbours affected"],
    "notes": "",
}


def test_parses_clean_json():
    assert parse_body(json.dumps(GOOD)).title.startswith("Evening")


def test_parses_json_inside_code_fences():
    wrapped = "```json\n" + json.dumps(GOOD) + "\n```"
    assert len(parse_body(wrapped).steps) == 5


def test_rejects_too_few_steps():
    bad = {**GOOD, "steps": GOOD["steps"][:2]}
    with pytest.raises(ValidationError):
        parse_body(json.dumps(bad))


def test_rejects_non_json():
    with pytest.raises(ValueError):
        parse_body("Sorry, I can't help with that.")

def test_lookalike_characters_are_normalized():
    data = {**GOOD, "title": "Wi\u2011Fi drops"}
    assert parse_body(json.dumps(data)).title == "Wi-Fi drops"