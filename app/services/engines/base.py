from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatResult:
    content: str
    model: str
    finish_reason: str
    usage: Usage | None


@dataclass
class TextDelta:
    content: str


@dataclass
class StreamEnd:
    model: str
    finish_reason: str
    usage: Usage | None


class Engine(ABC):
    """An LLM backend reachable over an OpenAI-compatible (or compatible-enough) HTTP API."""

    name: str
    model: str

    @abstractmethod
    def chat(self, messages: list[ChatMessage], **params) -> ChatResult: ...

    @abstractmethod
    def stream_chat(
        self, messages: list[ChatMessage], **params
    ) -> Iterator[TextDelta | StreamEnd]: ...
