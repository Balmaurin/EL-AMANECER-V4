"""
Exception handling and retry mechanisms for RAG system.

This module provides:
- Custom exception hierarchy
- Retry decorators with exponential backoff
- Circuit breaker pattern implementation
- Context managers for resources
- Structured error logging
"""

import functools
import logging
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, Optional, Type, Union

log = logging.getLogger("rag.errors")


# Base Exceptions
class RagError(Exception):
    """Base exception for all RAG system errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(message)


class ConfigError(RagError):
    """Configuration related errors."""

    pass


class DataError(RagError):
    """Data processing and validation errors."""

    pass


class IndexError(RagError):
    """Index operations errors."""

    pass


class ResourceError(RagError):
    """Resource allocation and management errors."""

    pass


class APIError(RagError):
    """API and network related errors."""

    pass


class SearchError(RagError):
    """Search operation errors."""

    pass


class CacheError(RagError):
    """Cache operation errors."""

    pass


class QueryExpansionError(RagError):
    """Query expansion operation errors."""

    pass


class FeedbackError(RagError):
    """Feedback operation errors."""

    pass


class ChunkQualityError(RagError):
    """Chunk quality errors."""

    pass


class ChunkingError(RagError):
    """Chunking operation errors."""

    pass


# Specific Exceptions
class ValidationError(DataError):
    """Data validation errors."""

    pass


class ConnectionError(ResourceError):
    """Database and external service connection errors."""

    pass


class AuthenticationError(APIError):
    """Authentication and authorization errors."""

    pass


class CircuitBreakerError(ResourceError):
    """Errors raised when circuit breaker is open."""

    pass


# Circuit Breaker
class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout

        self.failures = 0
        self.last_failure = 0
        self.state = "closed"
        self.lock = Lock()

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.lock:
                if self.state == "open":
                    if time.time() - self.last_failure > self.reset_timeout:
                        self.state = "half-open"
                    else:
                        raise CircuitBreakerError(
                            f"Circuit breaker is open for {func.__name__}"
                        )

            try:
                result = func(*args, **kwargs)

                with self.lock:
                    if self.state == "half-open":
                        self.state = "closed"
                        self.failures = 0

                return result

            except Exception as e:
                with self.lock:
                    self.failures += 1
                    self.last_failure = time.time()

                    if self.failures >= self.failure_threshold:
                        self.state = "open"

                    log.error(
                        f"Circuit breaker: {func.__name__} failed ({self.failures}/{self.failure_threshold})"
                    )
                raise

        return wrapper


# Retry Decorator
def retry(
    retries: int = 3,
    backoff_base: float = 2.0,
    backoff_max: float = 60.0,
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
) -> Callable[..., Any]:
    """Retry decorator with exponential backoff.

    Args:
        retries: Maximum number of retries
        backoff_base: Base for exponential backoff
        backoff_max: Maximum backoff time
        exceptions: Exception types to catch

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error = None

            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt == retries:
                        log.error(
                            f"Failed after {retries} retries: {func.__name__}",
                            exc_info=True,
                        )
                        raise

                    # Calculate backoff time
                    backoff = min(backoff_base**attempt, backoff_max)

                    log.warning(
                        f"Retry {attempt + 1}/{retries} for {func.__name__} "
                        f"after {backoff:.1f}s: {str(e)}"
                    )

                    time.sleep(backoff)

            if last_error:
                raise last_error

        return wrapper

    return decorator


# Resource Context Managers
@contextmanager
def resource_context(
    resource_type: str, resource_id: str, timeout: Optional[float] = None
):
    """Context manager for resource handling.

    Args:
        resource_type: Type of resource (db, file, etc)
        resource_id: Resource identifier
        timeout: Optional timeout in seconds

    Raises:
        ResourceError: If resource cannot be acquired
    """
    start = time.time()

    try:
        log.debug(f"Acquiring {resource_type}: {resource_id}")
        yield

    except Exception as e:
        log.error(f"Error in {resource_type} {resource_id}: {str(e)}", exc_info=True)
        raise ResourceError(
            f"Resource error: {resource_type} {resource_id}",
            details={"error": str(e), "traceback": traceback.format_exc()},
        )

    finally:
        duration = time.time() - start
        log.debug(
            f"Released {resource_type}: {resource_id} " f"(duration: {duration:.3f}s)"
        )

        if timeout and duration > timeout:
            log.warning(
                f"Resource usage exceeded timeout: {resource_type} "
                f"{resource_id} ({duration:.3f}s > {timeout:.3f}s)"
            )


# Error Logging
def log_error(
    error: Exception, context: Optional[Dict[str, Any]] = None, level: str = "error"
) -> None:
    """Log error with context and stack trace.

    Args:
        error: Exception to log
        context: Additional context information
        level: Log level to use
    """
    context = context or {}

    error_info = {
        "error_type": error.__class__.__name__,
        "message": str(error),
        "timestamp": datetime.now().isoformat(),
        "context": context,
        "traceback": traceback.format_exc(),
    }

    if isinstance(error, RagError):
        error_info.update(error.details)

    getattr(log, level.lower())(
        f"{error_info['error_type']}: {error_info['message']}",
        extra={"error_info": error_info},
    )
