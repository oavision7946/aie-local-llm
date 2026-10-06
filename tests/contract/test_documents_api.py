def test_ingest_then_list_document(client):
    response = client.post(
        "/documents", json={"content": "Some short fact.", "corpus": "testcorpus"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_created"] == 1
    doc_id = body["doc_id"]

    listed = client.get("/documents", params={"corpus": "testcorpus"})
    assert listed.status_code == 200
    docs = listed.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["doc_id"] == doc_id


def test_ingest_empty_content_rejected(client):
    response = client.post("/documents", json={"content": ""})
    assert response.status_code == 422


def test_delete_document_removes_chunks(client):
    created = client.post("/documents", json={"content": "Delete me.", "corpus": "testcorpus"})
    doc_id = created.json()["doc_id"]

    deleted = client.delete(f"/documents/{doc_id}")
    assert deleted.status_code == 200
    assert deleted.json()["chunks_removed"] == 1

    listed = client.get("/documents", params={"corpus": "testcorpus"})
    assert listed.json()["documents"] == []


def test_chat_with_rag_retrieves_ingested_document(client, fake_engine):
    client.post(
        "/documents", json={"content": "Our refund policy is 30 days.", "corpus": "policies"}
    )
    fake_engine.script("Refunds are allowed within 30 days.")

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "What is the refund policy?"}],
            "rag": {"enabled": True, "corpus": "policies"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) == 1
    assert "refund policy" in body["sources"][0]["text"]
