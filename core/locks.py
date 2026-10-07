from contextlib import contextmanager
import logging
from typing import Generator
from django.conf import settings

logger = logging.getLogger(__name__)


@contextmanager
def redis_lock(key: str, timeout: int = 60, blocking_timeout: int = 5) -> Generator[bool, None, None]:
    """
    Context manager acquiring a distributed lock using Redis.
    Falls back gracefully if Redis is unavailable in local development.
    """
    client = None
    lock = None
    try:
        import redis
        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        client = redis.Redis.from_url(redis_url)
        lock = client.lock(f"lock:{key}", timeout=timeout, sleep=0.1)
        acquired = lock.acquire(blocking=True, blocking_timeout=blocking_timeout)
        yield acquired
    except Exception as e:
        logger.debug(f"Redis lock unavailable for {key}: {e}. Proceeding in local fallback mode.")
        yield True
    finally:
        if lock and client:
            try:
                lock.release()
            except Exception:
                pass
