class EngineError(Exception):
    """Base error for engine failures.

    `safe_message` is user-facing; exception details are logged server-side only.
    """

    safe_message = "The local model backend is unavailable. Please try again shortly."


class TransientEngineError(EngineError):
    """A retryable failure (timeout, connection error, 5xx)."""


class EngineNotFoundError(EngineError):
    safe_message = "The requested model/engine is not configured."
