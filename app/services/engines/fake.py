from collections.abc import Iterator

from app.services.engines.base import (
    ChatMessage,
    ChatResult,
    Engine,
    StreamEnd,
    TextDelta,
    Usage,
)


class FakeEngine(Engine):
    """Deterministic, scriptable, no-network stand-in for tests."""

    def __init__(self, name: str = "fake", model: str = "fake-model") -> None:
        self.name = name
        self.model = model
        self._responses: list[str] = []
        self.received_messages: list[list[ChatMessage]] = []

    def script(self, *responses: str) -> "FakeEngine":
        self._responses = list(responses)
        return self

    def _next_response(self) -> str:
        if self._responses:
            return self._responses.pop(0)
        return "This is a fake response."

    def chat(self, messages: list[ChatMessage], **params) -> ChatResult:
        self.received_messages.append(messages)
        content = self._next_response()
        return ChatResult(
            content=content,
            model=self.model,
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=len(content.split()), total_tokens=10),
        )

    def stream_chat(
        self, messages: list[ChatMessage], **params
    ) -> Iterator[TextDelta | StreamEnd]:
        self.received_messages.append(messages)
        content = self._next_response()
        for word in content.split(" "):
            yield TextDelta(content=word + " ")
        yield StreamEnd(model=self.model, finish_reason="stop", usage=None)
