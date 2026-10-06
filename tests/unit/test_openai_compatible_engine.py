import json

import httpx
import openai
import pytest

from app.services.engines.base import ChatMessage
from app.services.engines.errors import EngineError, TransientEngineError
from app.services.engines.openai_compatible import OpenAICompatibleEngine


def make_engine(handler) -> OpenAICompatibleEngine:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = openai.OpenAI(base_url="http://fake-engine/v1", api_key="x", http_client=http_client)
    return OpenAICompatibleEngine(
        name="test", base_url="http://fake-engine/v1", model="m", client=client
    )


def test_chat_returns_parsed_result():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi there"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    engine = make_engine(handler)
    result = engine.chat([ChatMessage(role="user", content="hello")])

    assert result.content == "hi there"
    assert result.finish_reason == "stop"
    assert result.usage.total_tokens == 7


def test_chat_translates_5xx_to_transient_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    engine = make_engine(handler)
    with pytest.raises(TransientEngineError):
        engine.chat([ChatMessage(role="user", content="hello")])


def test_chat_translates_4xx_to_engine_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    engine = make_engine(handler)
    with pytest.raises(EngineError):
        engine.chat([ChatMessage(role="user", content="hello")])


def test_stream_chat_yields_text_deltas_then_end():
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [
            {
                "id": "chatcmpl-3",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": "hel"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-3",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
            },
        ]
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    from app.services.engines.base import StreamEnd, TextDelta

    engine = make_engine(handler)
    events = list(engine.stream_chat([ChatMessage(role="user", content="hi")]))

    deltas = [e for e in events if isinstance(e, TextDelta)]
    ends = [e for e in events if isinstance(e, StreamEnd)]
    assert "".join(d.content for d in deltas) == "hello"
    assert ends[-1].finish_reason == "stop"


def test_retries_transient_errors_then_succeeds():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 2:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": None,
            },
        )

    engine = make_engine(handler)
    result = engine.chat([ChatMessage(role="user", content="hello")])
    assert result.content == "ok"
    assert attempts["count"] == 2
