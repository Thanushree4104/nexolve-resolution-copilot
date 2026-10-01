import pytest

from app.core.taxonomy import load_taxonomy


def test_taxonomy_loads():
    tax = load_taxonomy()
    assert len(tax.classes) == 10


def test_held_back_classes_are_not_active():
    tax = load_taxonomy()
    active_ids = {c.id for c in tax.active_classes()}
    assert "tv.streaming" not in active_ids
    assert "security.fraud" not in active_ids
    assert len(active_ids) == 8


def test_every_class_has_enough_root_causes():
    for c in load_taxonomy().classes:
        assert len(c.root_causes) >= 3, c.id


def test_get_returns_class_or_none():
    tax = load_taxonomy()
    assert tax.get("billing.dispute").name == "Billing dispute"
    assert tax.get("does.not.exist") is None


def test_duplicate_ids_are_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 1\nproducts: [P]\nclasses:\n"
        "  - {id: a, name: A, description: d, products: [P]}\n"
        "  - {id: a, name: B, description: d, products: [P]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_taxonomy(bad)


def test_unknown_product_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 1\nproducts: [P]\nclasses:\n"
        "  - {id: a, name: A, description: d, products: [Nonexistent]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_taxonomy(bad)

def test_misspelled_field_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 1\nproducts: [P]\nclasses:\n"
        "  - {id: a, name: A, description: d, products: [P], root_cause: [x]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_taxonomy(bad)