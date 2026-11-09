from fastapi.testclient import TestClient


def test_health_returns_active_engine(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["active_engine"] == "fake"
