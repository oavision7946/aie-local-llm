import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_chat_service
from app.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    RetrievedSource,
    UsageOut,
)
from app.services.chat.service import ChatService, new_completion_id, now_epoch
from app.services.engines.base import ChatMessage, StreamEnd, TextDelta

router = APIRouter()


def _gen_params(request: ChatCompletionRequest) -> dict:
    params = {}
    if request.temperature is not None:
        params["temperature"] = request.temperature
    if request.max_tokens is not None:
        params["max_tokens"] = request.max_tokens
    return params


@router.post("/v1/chat/completions", response_model=None)
def chat_completions(
    request: ChatCompletionRequest, chat_service: ChatService = Depends(get_chat_service)
):
    messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]

    if request.stream:
        return StreamingResponse(
            _stream_response(chat_service, request, messages),
            media_type="text/event-stream",
        )

    outcome = chat_service.complete(
        request.model,
        messages,
        rag_enabled=request.rag.enabled,
        rag_corpus=request.rag.corpus,
        rag_top_k=request.rag.top_k,
        **_gen_params(request),
    )
    usage = None
    if outcome.usage is not None:
        usage = UsageOut(
            prompt_tokens=outcome.usage.prompt_tokens,
            completion_tokens=outcome.usage.completion_tokens,
            total_tokens=outcome.usage.total_tokens,
        )
    return ChatCompletionResponse(
        id=new_completion_id(),
        created=now_epoch(),
        model=outcome.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionChoiceMessage(content=outcome.content),
                finish_reason=outcome.finish_reason,
            )
        ],
        usage=usage,
        sources=[
            RetrievedSource(text=sc.chunk.text, score=sc.score, metadata=sc.chunk.metadata)
            for sc in outcome.sources
        ],
    )


def _stream_response(chat_service: ChatService, request: ChatCompletionRequest, messages):
    completion_id = new_completion_id()
    created = now_epoch()

    for event in chat_service.stream(
        request.model,
        messages,
        rag_enabled=request.rag.enabled,
        rag_corpus=request.rag.corpus,
        rag_top_k=request.rag.top_k,
        **_gen_params(request),
    ):
        if isinstance(event, TextDelta):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": {"content": event.content}, "finish_reason": None}
                ],
            }
        elif isinstance(event, StreamEnd):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": event.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": event.finish_reason}],
            }
        else:
            continue
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
