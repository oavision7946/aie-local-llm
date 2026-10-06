from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.deps import get_engine_registry
from app.services.engines.errors import EngineError
from app.services.engines.registry import EngineRegistry

router = APIRouter()


@router.get("/health")
def health(registry: EngineRegistry = Depends(get_engine_registry)):
    try:
        engine = registry.active()
    except EngineError:
        return JSONResponse(
            status_code=503, content={"status": "degraded", "reason": "no active engine"}
        )
    return {"status": "ok", "active_engine": engine.name, "model": engine.model}
