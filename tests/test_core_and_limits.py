import pytest
from django.core.exceptions import ValidationError
from django.contrib.auth import get_user_model
from apps.users.models import Plan
from apps.portfolios.models import Portfolio
from apps.brokers.models import BrokerAccount
from apps.strategies.models import Strategy, StrategyVersion
from apps.strategies.signals.catalog import STRATEGY_MAP
from apps.strategies.backtest import BacktestEngine
from core.security_guard import allow_local_writes, is_local_mode_active, ReadOnlyLocalEnvironmentError
import pandas as pd
import numpy as np

User = get_user_model()


@pytest.mark.django_db
def test_local_mode_read_only_protection():
    """Validates that LOCAL_MODE strictly prevents unauthorized writes on protected models."""
    if not is_local_mode_active():
        pytest.skip("LOCAL_MODE not active in environment")

    # Attempting to mutate a protected model without allow_local_writes must fail
    with pytest.raises(ReadOnlyLocalEnvironmentError):
        User.objects.create(email="forbidden_write@test.com", username="forbidden")

    # Reading should succeed without errors
    count = User.objects.count()
    assert count >= 0


@pytest.mark.django_db
def test_portfolio_limits_enforcement():
    """Validates that user plan limits are strictly enforced."""
    with allow_local_writes():
        plan, _ = Plan.objects.get_or_create(
            code="test_plan",
            defaults={"max_live_portfolios": 1, "max_paper_portfolios": 3}
        )
        user = User.objects.create(email="trader@test.com", username="trader", plan=plan)

        # 1. Create first live portfolio -> Success
        p1 = Portfolio.objects.create(user=user, name="Live 1", kind="live")
        assert p1.id is not None

        # 2. Attempt second live portfolio -> Must fail validation
        with pytest.raises(ValidationError):
            p2 = Portfolio(user=user, name="Live 2", kind="live")
            p2.full_clean()
            p2.save()

        # 3. Create 3 paper portfolios -> Success
        for i in range(3):
            Portfolio.objects.create(user=user, name=f"Paper {i+1}", kind="paper")

        # 4. Attempt 4th paper portfolio -> Must fail validation
        with pytest.raises(ValidationError):
            p_paper_extra = Portfolio(user=user, name="Paper 4", kind="paper")
            p_paper_extra.full_clean()
            p_paper_extra.save()


@pytest.mark.django_db
def test_paper_portfolio_broker_environment_constraints():
    """Validates that paper portfolios accept paper broker accounts and reject live broker accounts."""
    with allow_local_writes():
        user = User.objects.create(email="sim@test.com", username="sim")
        paper_port = Portfolio.objects.create(user=user, name="My Paper", kind="paper")

        # 1. Attaching paper broker account -> Success
        ba_paper = BrokerAccount(
            portfolio=paper_port,
            provider="alpaca",
            environment="paper",
            display_name="Alpaca Paper Test",
        )
        ba_paper.full_clean()
        ba_paper.save()
        assert ba_paper.id is not None

        # 2. Attaching live broker account to paper portfolio -> Must fail
        with pytest.raises(ValidationError):
            ba_live = BrokerAccount(
                portfolio=paper_port,
                provider="alpaca",
                environment="live",
                display_name="Alpaca Live Test",
            )
            ba_live.full_clean()
            ba_live.save()


def test_centralized_signal_catalog_parity():
    """Validates that the centralized strategy catalog returns valid signals and indicators."""
    assert "MeanRev_WeakRSI" in STRATEGY_MAP
    assert "bollinger_bands" in STRATEGY_MAP
    assert "TrendFollowing_GoldCross" in STRATEGY_MAP

    strat_func = STRATEGY_MAP["bollinger_bands"]
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    df = pd.DataFrame({
        "Open": np.linspace(100, 110, 50),
        "High": np.linspace(101, 111, 50),
        "Low": np.linspace(99, 109, 50),
        "Close": np.linspace(100, 110, 50),
        "Volume": 100000.0,
    }, index=dates)

    signals, indicators = strat_func(df, [20, 2.0])
    assert len(signals) == 50
    assert isinstance(indicators, pd.DataFrame)


def test_backtest_engine_execution():
    """Validates that BacktestEngine runs using the same centralized catalog."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    prices = 100.0 + (pd.Series(range(100)) * 0.2) + (pd.Series(range(100)).apply(lambda x: (x % 5) - 2))
    df = pd.DataFrame({
        "Open": prices,
        "High": prices + 1.0,
        "Low": prices - 1.0,
        "Close": prices,
        "Volume": 500000.0,
    }, index=dates)

    engine = BacktestEngine(initial_cash=50000.0)
    res = engine.run(df, "bollinger_bands", [20, 2.0])

    assert "final_value" in res
    assert "roi_pct" in res
    assert "sharpe_ratio" in res
    assert "equity_curve" in res
    assert len(res["equity_curve"]) > 0


@pytest.mark.django_db
def test_bot_instance_creation_and_cross_asset():
    """Validates creating OneStrategy, MultiStrategy and CrossAsset Bot instances with correct roles."""
    from apps.strategies.serializers import BotCreateSerializer
    from apps.strategies.models import StrategyInstrument

    with allow_local_writes():
        # 1. Create OneStrategy Bot
        s1 = BotCreateSerializer(data={
            "name": "TQQQ Bollinger Bot",
            "bot_type": "one_strategy",
            "traded_symbol": "TQQQ",
            "strategies": [{"strategy_name": "MeanRev_BollingerBands", "params": [20, 2.0]}],
        })
        assert s1.is_valid(), s1.errors
        bot1 = s1.save()
        assert bot1.kind == "one_strategy"
        version1 = bot1.versions.first()
        assert version1.instruments.filter(role="traded").first().instrument.symbol == "TQQQ"

        # 2. Create CrossAsset Bot
        s2 = BotCreateSerializer(data={
            "name": "Bond Rally TQQQ Cross Bot",
            "bot_type": "cross_asset",
            "traded_symbol": "TQQQ",
            "signal_symbol": "TLT",
            "strategies": [{"strategy_name": "CrossAssets_BondsRallying", "params": [20, 1.5]}],
        })
        assert s2.is_valid(), s2.errors
        bot2 = s2.save()
        assert bot2.kind == "cross_asset"
        version2 = bot2.versions.first()
        assert version2.instruments.filter(role="traded").first().instrument.symbol == "TQQQ"
        assert version2.instruments.filter(role="signal_source").first().instrument.symbol == "TLT"


@pytest.mark.django_db
def test_portfolio_drift_calculation():
    """Validates portfolio-level rebalance tolerance and drift metrics calculation."""
    from apps.portfolios.models import Allocation
    from decimal import Decimal

    with allow_local_writes():
        user = User.objects.create(email="drift_user@test.com", username="drift_user")
        portfolio = Portfolio.objects.create(
            user=user,
            name="Macro Portfolio",
            rebalance_frequency="daily",
            rebalance_tolerance=Decimal("0.1000"),
            rebalance_tolerance_mode="relative",
            paper_initial_capital=Decimal("100000.00")
        )
        strat = Strategy.objects.create(name="Dummy Strat", slug="dummy-strat", kind="one_strategy")
        Allocation.objects.create(portfolio=portfolio, strategy=strat, target_weight=Decimal("0.5000"), enabled=True)

        drift_data = portfolio.calculate_drift_metrics()
        assert drift_data["portfolio_id"] == portfolio.id
        assert drift_data["rebalance_frequency"] == "daily"
        assert len(drift_data["allocations"]) == 1
        alloc_metric = drift_data["allocations"][0]
        assert alloc_metric["target_weight"] == 0.5
        # Since current value is 0, drift is breached
        assert alloc_metric["breached"] is True


@pytest.mark.django_db
def test_broker_credential_encryption():
    """Validates that broker credentials encrypt and decrypt properly with Fernet."""
    from apps.brokers.models import BrokerCredential

    with allow_local_writes():
        user = User.objects.create(email="crypto_user@test.com", username="crypto_user")
        cred = BrokerCredential.objects.create(
            user=user,
            provider="alpaca",
            environment="paper",
        )
        secrets = {"key_id": "PKTEST123456", "secret_key": "SECRET_TEST_9999"}
        cred.set_secrets(secrets)
        cred.save()

        # Re-fetch from database to ensure decrypt works
        fetched = BrokerCredential.objects.get(id=cred.id)
        decrypted = fetched.get_secrets()
        assert decrypted["key_id"] == "PKTEST123456"
        assert decrypted["secret_key"] == "SECRET_TEST_9999"
        assert fetched.key_fingerprint == "3456"

