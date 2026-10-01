from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_response_has_request_id():
    response = client.get("/health")
    assert response.headers["X-Request-ID"]


def test_incoming_request_id_is_echoed():
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"