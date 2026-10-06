import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.services.engines.base import ChatMessage, StreamEnd, TextDelta
from app.services.engines.registry import EngineRegistry
from app.services.rag.service import RagService
from app.services.rag.vector_store.base import ScoredChunk


@dataclass
class ChatOutcome:
    content: str
    model: str
    finish_reason: str
    usage: object | None
    sources: list[ScoredChunk] = field(default_factory=list)


class ChatService:
    def __init__(
        self,
        registry: EngineRegistry,
        rag_service: RagService,
        *,
        default_top_k: int,
        default_corpus: str,
    ) -> None:
        self._registry = registry
        self._rag_service = rag_service
        self._default_top_k = default_top_k
        self._default_corpus = default_corpus

    def _augment_with_rag(
        self, messages: list[ChatMessage], *, enabled: bool, corpus: str | None, top_k: int | None
    ) -> tuple[list[ChatMessage], list[ScoredChunk]]:
        if not enabled:
            return messages, []

        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        if last_user is None:
            return messages, []

        sources = self._rag_service.retrieve(
            last_user.content,
            corpus=corpus or self._default_corpus,
            top_k=top_k or self._default_top_k,
        )
        if not sources:
            return messages, []

        context_message = ChatMessage(
            role="system", content=self._rag_service.format_context(sources)
        )
        return [context_message, *messages], sources

    def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        rag_enabled: bool = False,
        rag_corpus: str | None = None,
        rag_top_k: int | None = None,
        **params,
    ) -> ChatOutcome:
        engine = self._registry.resolve(model)
        augmented, sources = self._augment_with_rag(
            messages, enabled=rag_enabled, corpus=rag_corpus, top_k=rag_top_k
        )
        result = engine.chat(augmented, **params)
        return ChatOutcome(
            content=result.content,
            model=result.model,
            finish_reason=result.finish_reason,
            usage=result.usage,
            sources=sources,
        )

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        rag_enabled: bool = False,
        rag_corpus: str | None = None,
        rag_top_k: int | None = None,
        **params,
    ) -> Iterator[TextDelta | StreamEnd]:
        engine = self._registry.resolve(model)
        augmented, _sources = self._augment_with_rag(
            messages, enabled=rag_enabled, corpus=rag_corpus, top_k=rag_top_k
        )
        yield from engine.stream_chat(augmented, **params)


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def now_epoch() -> int:
    return int(time.time())
