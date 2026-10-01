import pytest

from app.core.pii import redact
from app.llm.mock import MockProvider
from app.llm.redacting import RedactingProvider


def test_email_is_redacted():
    r = redact("write to ravi.kumar@example.com please")
    assert r.text == "write to <EMAIL> please"


@pytest.mark.parametrize(
    "number",
    ["9876543210", "98765 43210", "+91 98765 43210", "+91-9876543210", "09876543210"],
)
def test_indian_phone_formats(number):
    r = redact(f"call me on {number} tonight")
    assert "<PHONE>" in r.text
    assert "98765" not in r.text


def test_german_phone_is_redacted():
    r = redact("Rufen Sie mich unter +49 170 1234567 an")
    assert "<PHONE>" in r.text
    assert "1234567" not in r.text


def test_aadhaar_is_redacted():
    assert "<AADHAAR>" in redact("my id is 1234 5678 9012").text


def test_pan_is_redacted():
    assert "<PAN>" in redact("PAN ABCDE1234F on file").text


def test_account_number_keeps_label():
    r = redact("My account number 12345678 is locked")
    assert r.text == "My account number <ACCOUNT> is locked"


def test_clean_text_is_unchanged():
    text = "My broadband drops every evening around 8 and I restarted the router twice. Error E-203 on Nimbus ONT-5G."
    r = redact(text)
    assert r.text == text
    assert not r.found


def test_existing_dataset_placeholders_untouched():
    text = "Dear <name>, call <tel_num> about <acc_num>"
    assert redact(text).text == text


def test_counts_are_reported():
    r = redact("a@b.com and c@d.com, call 9876543210")
    assert r.counts == {"EMAIL": 2, "PHONE": 1}


def test_llm_never_receives_pii():
    mock = MockProvider()
    provider = RedactingProvider(mock)
    provider.complete(
        [{"role": "user", "content": "Call me on +91 98765 43210 or mail ravi@example.com"}]
    )
    sent = str(mock.calls)
    assert "98765" not in sent
    assert "ravi@example.com" not in sent
    assert "<PHONE>" in sent
    assert "<EMAIL>" in sent