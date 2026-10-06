from collections.abc import Iterator

import httpx
import openai

from app.services.engines.base import (
    ChatMessage,
    ChatResult,
    Engine,
    StreamEnd,
    TextDelta,
    Usage,
)
from app.services.engines.errors import EngineError, TransientEngineError
from app.services.engines.retry import call_with_retry

TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class OpenAICompatibleEngine(Engine):
    """Talks to any server exposing the OpenAI chat completions API (vLLM, SGLang, LMCache-vLLM)."""

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        *,
        api_key: str = "not-needed",
        client: openai.OpenAI | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._client = client or openai.OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages: list[ChatMessage], **params) -> ChatResult:
        return call_with_retry(lambda: self._chat_once(messages, **params))

    def _chat_once(self, messages: list[ChatMessage], **params) -> ChatResult:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                stream=False,
                **params,
            )
        except Exception as exc:
            raise _translate_error(exc) from exc

        choice = response.choices[0]
        usage = None
        if response.usage is not None:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        return ChatResult(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )

    def stream_chat(
        self, messages: list[ChatMessage], **params
    ) -> Iterator[TextDelta | StreamEnd]:
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                stream=True,
                **params,
            )
            model_name = self.model
            finish_reason = "stop"
            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                model_name = chunk.model or model_name
                if choice and choice.delta and choice.delta.content:
                    yield TextDelta(content=choice.delta.content)
                if choice and choice.finish_reason:
                    finish_reason = choice.finish_reason
            yield StreamEnd(model=model_name, finish_reason=finish_reason, usage=None)
        except Exception as exc:
            raise _translate_error(exc) from exc


def _translate_error(exc: Exception) -> EngineError:
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
        return TransientEngineError(str(exc))
    if isinstance(exc, openai.APIStatusError) and exc.status_code in TRANSIENT_STATUS_CODES:
        return TransientEngineError(str(exc))
    if isinstance(exc, openai.APIConnectionError | openai.APITimeoutError):
        return TransientEngineError(str(exc))
    return EngineError(str(exc))
