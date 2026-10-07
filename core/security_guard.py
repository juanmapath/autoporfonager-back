import os
from contextlib import contextmanager
from typing import Generator
import threading
from django.conf import settings
from django.core.exceptions import PermissionDenied

_local_override = threading.local()

# Models strictly protected from local mutations (live client trades, real orders, broker credentials)
PROTECTED_MODELS = {
    "Order",
    "Fill",
    "FillAllocation",
    "OrderIntent",
    "RebalanceRun",
    "Target",
    "LogicalPosition",
    "BrokerPositionSnapshot",
    "LedgerEntry",
    "EquitySnapshot",
    "SleevePnlDaily",
    "BrokerCredential",
    "KillSwitch",
    "FundingAlert",
}


class ReadOnlyLocalEnvironmentError(PermissionDenied):
    """Raised when an attempt is made to mutate protected client or trading data while in LOCAL_MODE."""
    pass


def is_local_mode_active() -> bool:
    """Checks whether LOCAL_MODE is enabled via settings or environment variable."""
    env_val = os.getenv("LOCAL_MODE", os.getenv("LOCAL", "")).strip().lower()
    if env_val in ("true", "1", "yes", "t"):
        return True
    if env_val in ("false", "0", "no", "f"):
        return False
    return getattr(settings, "LOCAL_MODE", False)


def is_local_write_allowed() -> bool:
    """Checks whether a local thread-specific override is currently active (e.g. during migrations or test fixtures)."""
    return getattr(_local_override, "allow_writes", False)


@contextmanager
def allow_local_writes() -> Generator[None, None, None]:
    """Context manager to allow database mutations in local environment (used for tests or migrations)."""
    prev = getattr(_local_override, "allow_writes", False)
    _local_override.allow_writes = True
    try:
        yield
    finally:
        _local_override.allow_writes = prev


class LocalModeSecurityRouter:
    """
    Database router that blocks INSERT, UPDATE, DELETE operations on protected
    client and trading tables when LOCAL_MODE is active, guaranteeing that
    local environments can only perform SELECT / GET operations against production or replica databases.
    """

    def db_for_read(self, model, **hints):
        return "default"

    def db_for_write(self, model, **hints):
        if is_local_mode_active() and not is_local_write_allowed():
            if model.__name__ in PROTECTED_MODELS:
                raise ReadOnlyLocalEnvironmentError(
                    f"LOCAL_MODE is ACTIVE! Mutation (write/update/delete) on protected model "
                    f"'{model._meta.app_label}.{model.__name__}' is strictly prohibited in local environment. "
                    f"Only read (GET) queries and local research scripts are allowed."
                )
        return "default"

    def allow_relation(self, obj1, obj2, **hints):
        return True

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        return True
