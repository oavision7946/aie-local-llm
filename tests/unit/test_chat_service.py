from app.services.chat.service import ChatService
from app.services.engines.base import ChatMessage
from app.services.engines.fake import FakeEngine
from app.services.engines.registry import EngineRegistry
from app.services.rag.embeddings.fake import FakeEmbedder
from app.services.rag.service import RagService
from app.services.rag.vector_store.memory import MemoryVectorStore


def make_chat_service(engine):
    registry = EngineRegistry(engines={"fake": engine}, active_name="fake")
    rag_service = RagService(
        embedder=FakeEmbedder(), vector_store=MemoryVectorStore(), chunk_size=5000, chunk_overlap=0
    )
    return ChatService(
        registry=registry, rag_service=rag_service, default_top_k=3, default_corpus="default"
    ), rag_service


def test_complete_without_rag_sends_messages_unmodified():
    engine = FakeEngine().script("plain answer")
    chat_service, _ = make_chat_service(engine)

    outcome = chat_service.complete("fake", [ChatMessage(role="user", content="hello")])

    assert outcome.content == "plain answer"
    assert outcome.sources == []
    assert engine.received_messages[0] == [ChatMessage(role="user", content="hello")]


def test_complete_with_rag_prepends_context_and_returns_sources():
    engine = FakeEngine().script("answer with context")
    chat_service, rag_service = make_chat_service(engine)
    rag_service.ingest("Paris is the capital of France.", corpus="default")

    outcome = chat_service.complete(
        "fake",
        [ChatMessage(role="user", content="Paris is the capital of France.")],
        rag_enabled=True,
    )

    assert outcome.content == "answer with context"
    assert len(outcome.sources) == 1
    sent_messages = engine.received_messages[0]
    assert sent_messages[0].role == "system"
    assert "Paris" in sent_messages[0].content
    assert sent_messages[-1].role == "user"


def test_complete_with_rag_no_matches_does_not_add_system_message():
    engine = FakeEngine().script("no context needed")
    chat_service, _rag_service = make_chat_service(engine)

    outcome = chat_service.complete(
        "fake", [ChatMessage(role="user", content="anything")], rag_enabled=True
    )

    assert outcome.sources == []
    assert engine.received_messages[0] == [ChatMessage(role="user", content="anything")]


def test_stream_yields_text_deltas_then_end():
    engine = FakeEngine().script("hello world")
    chat_service, _ = make_chat_service(engine)

    events = list(chat_service.stream("fake", [ChatMessage(role="user", content="hi")]))

    from app.services.engines.base import StreamEnd, TextDelta

    assert isinstance(events[-1], StreamEnd)
    assert any(isinstance(e, TextDelta) for e in events)
