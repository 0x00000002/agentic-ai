import asyncio
import random
# Remove standard logging import
# import logging 
from functools import wraps
from typing import Callable, Any, Coroutine, TypeVar, ParamSpec, Tuple, Type

# Import the framework's logger factory and interface
from .logger import LoggerFactory, LoggerInterface

# Configure logger using the framework's factory
logger: LoggerInterface = LoggerFactory.create(__name__)

# Type variables for generic decorator
T = TypeVar('T')
P = ParamSpec('P')

DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF = 1.0  # seconds
DEFAULT_MAX_BACKOFF = 10.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_JITTER_FACTOR = 0.1 # Percentage of backoff time

def async_retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    jitter_factor: float = DEFAULT_JITTER_FACTOR,
    # Explicitly type the tuple elements
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,) # Default to retry on any Exception
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """
    Decorator to automatically retry an asynchronous function with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_backoff: Initial delay in seconds before the first retry.
        max_backoff: Maximum delay in seconds between retries.
        backoff_factor: Factor by which the delay increases (e.g., 2 for exponential backoff).
        jitter_factor: Percentage (0 to 1) of backoff time to add/subtract as random jitter.
        retry_on_exceptions: A tuple of exception types that should trigger a retry.

    Returns:
        A decorator function.
    """
    def decorator(func: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            retries = 0
            backoff_time = initial_backoff
            last_exception: Exception | None = None

            while retries <= max_retries:
                try:
                    # Await the coroutine returned by func
                    return await func(*args, **kwargs)
                except retry_on_exceptions as e:
                    last_exception = e
                    retries += 1
                    if retries > max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}. Last error: {e!r}",
                            exc_info=True # Include stack trace in log
                        )
                        raise # Reraise the last exception

                    # Calculate jitter: random value between -jitter and +jitter percentage of backoff
                    jitter = random.uniform(-jitter_factor, jitter_factor) * backoff_time
                    # Apply jitter and ensure wait time is non-negative
                    wait_time = max(0, backoff_time + jitter)

                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} after error: {e!r}. "
                        f"Waiting {wait_time:.2f}s..."
                        # Consider adding exc_info=True for more detailed logs on retries if needed
                    )
                    await asyncio.sleep(wait_time)

                    # Increase backoff time for the next retry, capped at max_backoff
                    backoff_time = min(max_backoff, backoff_time * backoff_factor)

            # This part should ideally not be reached if max_retries >= 0 and func eventually
            # returns or raises an exception caught by retry_on_exceptions.
            # If it's reached, it means the loop completed without returning or raising a caught exception.
            if last_exception:
                # If loop ended due to retries exhaustion, last_exception is set.
                 # This path is handled by the 'raise' inside the loop now.
                 # If we somehow exit the loop otherwise with a last_exception, re-raise it.
                raise last_exception
            else:
                # This scenario is highly unlikely if retry_on_exceptions is not empty
                # and the decorated function behaves as expected (returns or raises).
                # Raise a generic error indicating an unexpected state.
                raise RuntimeError(f"{func.__name__} failed after retries without a conclusive exception.")

        return wrapper
    return decorator

# --- Example Usage (Keep commented out) ---
# from src.exceptions import AIProviderError # Example specific exception

# # Example exceptions to retry on
# RETRYABLE_EXCEPTIONS = (
#     ConnectionError,
#     TimeoutError,
#     # Add provider-specific transient errors here, e.g.:
#     # SomeProviderRateLimitError,
#     # SomeProviderTemporaryServerError
# )

# @async_retry_with_backoff(max_retries=2, initial_backoff=0.5, retry_on_exceptions=RETRYABLE_EXCEPTIONS)
# async def potentially_flaky_api_call(url: str) -> dict:
#     # Simulate network request
#     print(f"Attempting to call {url}...")
#     if random.random() < 0.7: # Simulate failure 70% of the time
#         fail_type = random.choice([ConnectionError, TimeoutError, ValueError])
#         print(f"Simulating failure with {fail_type.__name__}")
#         raise fail_type("Simulated network/API issue")
#     print("Call successful!")
#     return {"data": "some data"}

# async def main():
#     try:
#         result = await potentially_flaky_api_call("http://example.com/api")
#         print(f"Final result: {result}")
#     except Exception as e:
#         # This will catch the exception after retries are exhausted or a non-retryable exception
#         print(f"API call failed definitely after retries: {e!r}")

# if __name__ == "__main__":
#      asyncio.run(main()) 