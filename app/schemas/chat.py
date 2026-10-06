from pydantic import BaseModel, Field


class RagOptions(BaseModel):
    enabled: bool = False
    corpus: str | None = None
    top_k: int | None = None


class ChatMessageIn(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI /v1/chat/completions request shape, extended with an opt-in `rag` block.

    Plain OpenAI clients that omit `rag` get standard passthrough chat behavior.
    """

    model: str
    messages: list[ChatMessageIn]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    rag: RagOptions = Field(default_factory=RagOptions)


class ChatCompletionChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: str


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RetrievedSource(BaseModel):
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageOut | None = None
    sources: list[RetrievedSource] = Field(default_factory=list)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "aie-local-llm"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
