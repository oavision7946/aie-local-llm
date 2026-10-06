def test_chat_completions_basic(client, fake_engine):
    fake_engine.script("hello back")

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hello back"
    assert body["sources"] == []


def test_chat_completions_with_rag_returns_sources(client, fake_engine, rag_service):
    fake_engine.script("answer using context")
    rag_service.ingest("The Eiffel Tower is in Paris.", corpus="default")

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "The Eiffel Tower is in Paris."}],
            "rag": {"enabled": True},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "answer using context"
    assert len(body["sources"]) == 1
    assert "Eiffel" in body["sources"][0]["text"]


def test_chat_completions_unknown_model_returns_502(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502


def test_chat_completions_rejects_invalid_payload(client):
    response = client.post("/v1/chat/completions", json={"model": "fake-model"})
    assert response.status_code == 422


def test_chat_completions_stream_emits_sse(client, fake_engine):
    fake_engine.script("stream me")

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "data: [DONE]" in body
    assert "stream" in body
