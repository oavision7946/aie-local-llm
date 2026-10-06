from fastapi import FastAPI

from app.api import chat, documents, health, models
from app.api.errors import register_exception_handlers

app = FastAPI(
    title="aie-local-llm",
    description="OpenAI-compatible local-model QA/RAG gateway",
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(documents.router)
