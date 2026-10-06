from fastapi import APIRouter, Depends

from app.api.deps import get_engine_registry
from app.schemas.chat import ModelInfo, ModelsResponse
from app.services.engines.registry import EngineRegistry

router = APIRouter()


@router.get("/v1/models", response_model=ModelsResponse)
def list_models(registry: EngineRegistry = Depends(get_engine_registry)) -> ModelsResponse:
    return ModelsResponse(data=[ModelInfo(id=engine.model) for engine in registry.list_engines()])
