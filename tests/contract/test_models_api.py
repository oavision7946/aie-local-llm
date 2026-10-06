def test_list_models_returns_configured_engine(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert any(m["id"] == "fake-model" for m in body["data"])
