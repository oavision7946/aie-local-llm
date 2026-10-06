import time
from collections.abc import Callable

from app.services.engines.errors import TransientEngineError


def call_with_retry[T](
    fn: Callable[[], T], *, max_attempts: int = 3, base_delay_s: float = 0.05
) -> T:
    last_error: TransientEngineError | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except TransientEngineError as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(base_delay_s * (2**attempt))
    assert last_error is not None
    raise last_error
