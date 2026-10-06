import os

import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:9000")


def list_models() -> list[str]:
    response = httpx.get(f"{API_BASE_URL}/v1/models", timeout=10)
    response.raise_for_status()
    return [m["id"] for m in response.json()["data"]]


def chat(
    model: str,
    messages: list[dict],
    *,
    rag_enabled: bool,
    corpus: str,
    top_k: int,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "rag": {"enabled": rag_enabled, "corpus": corpus, "top_k": top_k},
    }
    response = httpx.post(f"{API_BASE_URL}/v1/chat/completions", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def ingest_document(content: str, *, corpus: str) -> dict:
    response = httpx.post(
        f"{API_BASE_URL}/documents", json={"content": content, "corpus": corpus}, timeout=60
    )
    response.raise_for_status()
    return response.json()


def list_documents(corpus: str) -> list[dict]:
    response = httpx.get(f"{API_BASE_URL}/documents", params={"corpus": corpus}, timeout=10)
    response.raise_for_status()
    return response.json()["documents"]
