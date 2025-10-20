import os
import time
import random
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable


class RetryConfig:
    def __init__(self):
        self.max_retries = int(os.environ.get("AI_MAX_RETRIES", "3"))
        self.base_delay = float(os.environ.get("AI_BASE_DELAY", "1.0"))
        self.max_delay = float(os.environ.get("AI_MAX_DELAY", "30.0"))
        self.jitter_max = float(os.environ.get("AI_JITTER_MAX", "0.5"))
        self.exponential_base = float(os.environ.get("AI_EXPONENTIAL_BASE", "2.0"))


class CircuitBreakerState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.logger = logging.getLogger(__name__ + '.CircuitBreaker')

    def call(self, func: Callable, *args, **kwargs):
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - failing fast")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        if not self.last_failure_time:
            return False
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)

    def _on_success(self):
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.success_count = 0
                self.logger.info("Circuit breaker transitioning to CLOSED state")

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning("Circuit breaker transitioning back to OPEN state")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.error(f"Circuit breaker tripped! Transitioning to OPEN state after {self.failure_count} failures")


class AIServiceError(Exception):
    def __init__(self, message: str, error_type: str = "unknown", is_retryable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.is_retryable = is_retryable


from typing import Optional


def intelligent_retry(config: Optional[RetryConfig] = None, circuit_breaker: Optional[CircuitBreaker] = None):
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            retry_logger = logging.getLogger(f"{__name__}.retry.{func.__name__}")

            for attempt in range(config.max_retries + 1):
                try:
                    if circuit_breaker:
                        return circuit_breaker.call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    is_retryable = True
                    error_type = "unknown"

                    if isinstance(e, AIServiceError):
                        is_retryable = e.is_retryable
                        error_type = e.error_type
                    elif "quota" in str(e).lower() or "rate limit" in str(e).lower():
                        error_type = "rate_limit"
                        is_retryable = True
                    elif "api key" in str(e).lower() or "authentication" in str(e).lower():
                        error_type = "authentication"
                        is_retryable = False
                    elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                        error_type = "network"
                        is_retryable = True
                    elif "model" in str(e).lower() and "not found" in str(e).lower():
                        error_type = "model_not_found"
                        is_retryable = False

                    if attempt == 0:
                        retry_logger.warning(f"Initial attempt failed: {error_type} - {str(e)}")
                    else:
                        retry_logger.warning(f"Retry attempt {attempt}/{config.max_retries} failed: {error_type} - {str(e)}")

                    if not is_retryable or attempt >= config.max_retries:
                        retry_logger.error(f"Giving up after {attempt + 1} attempts. Last error: {str(e)}")
                        break

                    base_delay = config.base_delay * (config.exponential_base ** attempt)
                    capped_delay = min(base_delay, config.max_delay)
                    jitter = random.uniform(0, config.jitter_max)
                    total_delay = capped_delay + jitter

                    retry_logger.info(f"Retrying in {total_delay:.2f}s (base: {capped_delay:.2f}s + jitter: {jitter:.2f}s)")
                    time.sleep(total_delay)

            if isinstance(last_exception, AIServiceError):
                raise last_exception
            else:
                raise AIServiceError(
                    f"AI service call failed after {config.max_retries + 1} attempts: {str(last_exception)}",
                    error_type=error_type,
                    is_retryable=False
                )

        return wrapper
    return decorator
