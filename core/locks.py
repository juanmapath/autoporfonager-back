from contextlib import contextmanager
import logging
from typing import Generator
from django.conf import settings

logger = logging.getLogger(__name__)


def _redis_client():
    """
    Cliente Redis para locks; aislado para poder desactivarlo en tests. Timeouts cortos: si Redis
    no responde, el lock cae al modo local en ~1 s en vez de esperar el timeout TCP del sistema
    (decenas de segundos por portafolio dentro de la ventana de cierre).
    """
    import redis
    redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(redis_url, socket_connect_timeout=1, socket_timeout=2)


@contextmanager
def redis_lock(key: str, timeout: int = 60, blocking_timeout: int = 5) -> Generator[bool, None, None]:
    """
    Context manager acquiring a distributed lock using Redis.
    Falls back gracefully (yields True) if Redis is unavailable in local development: in that case
    the lock gives no mutual exclusion, so callers that need a hard guarantee must also serialize
    in the database (select_for_update).
    Only acquisition errors trigger the fallback; exceptions raised inside the `with` body propagate.
    """
    lock = None
    try:
        client = _redis_client()
        lock = client.lock(f"lock:{key}", timeout=timeout, sleep=0.1)
        acquired = lock.acquire(blocking=True, blocking_timeout=blocking_timeout)
    except Exception as e:
        logger.debug(f"Redis lock unavailable for {key}: {e}. Proceeding in local fallback mode.")
        lock = None
        acquired = True

    try:
        yield acquired
    finally:
        if lock is not None and acquired:
            try:
                lock.release()
            except Exception:
                pass
