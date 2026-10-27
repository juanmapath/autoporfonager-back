import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _trading_day_and_no_redis():
    """
    Tests deterministas sin importar la fecha real: el guard de día hábil ve hoy como sesión NYSE
    (los tests de feriado lo parchean a False) y los locks Redis usan el modo local (sin conexión,
    que en Windows tarda ~2 s por intento contra un puerto cerrado).
    """
    with patch("apps.trading.guards.is_trading_day", return_value=True), \
            patch("core.locks._redis_client", side_effect=ConnectionError("redis disabled in tests")):
        yield


def pytest_configure(config):
    config.addinivalue_line("markers", "postgres: requiere DATABASE_URL de Postgres (concurrencia real)")
