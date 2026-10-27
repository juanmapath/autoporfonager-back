import io
from unittest.mock import patch

import pytest
from django.core.management import call_command

from apps.instruments.models import Instrument
from apps.marketdata.models import Bar
from apps.portfolios.models import Portfolio
from apps.strategies.models import Signal, StrategyVersion
from apps.trading.models import DailyPipelineRun, Order, OrderIntent, RebalanceRun


@pytest.mark.django_db
def test_benchmark_reports_steps_and_persists_nothing():
    out = io.StringIO()
    with patch("alpaca.trading.client.TradingClient") as alpaca:
        call_command(
            "benchmark_pipeline", "--bots", "4", "--instruments", "12", "--instruments-per-bot", "5",
            "--no-network", "--latency-ms", "0", "--history-years", "2", stdout=out,
        )

    report = out.getvalue()
    assert "Benchmark pipeline: 4 bots, universo 12 instrumentos" in report
    for step in ("1. marketdata", "2. senales", "3. orquestacion", "despacho (simulado)", "Pipeline 1+2+3"):
        assert step in report
    assert "Senales por bot: p50" in report and "n=4" in report
    assert "CUMPLE" in report

    # Todo se revierte: ni bots, ni barras, ni señales, ni runs, ni órdenes
    alpaca.assert_not_called()
    for model in (Instrument, Bar, Signal, StrategyVersion, Portfolio, RebalanceRun, OrderIntent, Order, DailyPipelineRun):
        assert not model.objects.exists(), model.__name__
