from rest_framework import serializers
from decimal import Decimal
from apps.portfolios.models import Portfolio, Allocation, RecommendedPortfolio, RecommendedAllocation
from apps.brokers.serializers import BrokerAccountSerializer
from apps.strategies.serializers import StrategySerializer


class AllocationSerializer(serializers.ModelSerializer):
    strategy_name = serializers.CharField(source="strategy.name", read_only=True)
    strategy_slug = serializers.CharField(source="strategy.slug", read_only=True)
    strategy_detail = StrategySerializer(source="strategy", read_only=True)

    class Meta:
        model = Allocation
        fields = [
            "id", "portfolio", "strategy", "strategy_name", "strategy_slug", "strategy_detail",
            "target_weight", "broker_account", "enabled",
        ]
        read_only_fields = ["id"]


class PortfolioSerializer(serializers.ModelSerializer):
    allocations = AllocationSerializer(many=True, read_only=True)
    broker_accounts = BrokerAccountSerializer(many=True, read_only=True)
    current_value = serializers.SerializerMethodField()
    equity_source = serializers.SerializerMethodField()
    equity_synced_at = serializers.SerializerMethodField()
    initial_value = serializers.SerializerMethodField()
    net_contributions = serializers.SerializerMethodField()
    total_pnl = serializers.SerializerMethodField()
    total_return_pct = serializers.SerializerMethodField()
    primary_broker = serializers.SerializerMethodField()

    class Meta:
        model = Portfolio
        fields = [
            "id", "user", "kind", "name", "base_currency", "status",
            "trading_enabled", "rebalance_frequency", "rebalance_tolerance", "rebalance_tolerance_mode",
            "max_gross_allocation", "paper_initial_capital",
            "follows", "rebalance_pending", "allocations", "broker_accounts",
            "current_value", "equity_source", "equity_synced_at", "initial_value", "net_contributions", "total_pnl", "total_return_pct", "primary_broker",
            "created_at",
        ]
        read_only_fields = ["id", "user", "created_at"]

    def _equity(self, obj):
        # Saldo del broker si está sincronizado; si no, equity contable (sin refrescar cotizaciones).
        cache = self.__dict__.setdefault("_equity_cache", {})
        if obj.pk not in cache:
            from apps.accounting.services import portfolio_equity
            cache[obj.pk] = portfolio_equity(obj, refresh=False)
        return cache[obj.pk]

    def get_current_value(self, obj):
        return float(self._equity(obj)["equity"])

    def get_equity_source(self, obj):
        return self._equity(obj)["source"]

    def get_equity_synced_at(self, obj):
        return self._equity(obj)["synced_at"]

    def get_initial_value(self, obj):
        # Capital al crear el portafolio o al vincular el broker.
        return float(obj.paper_initial_capital)

    def get_net_contributions(self, obj):
        from apps.accounting.services import net_cash_flows
        return float(net_cash_flows(obj))

    def get_total_pnl(self, obj):
        # Ganancia real: lo que hay menos lo que se aportó (inicial + depósitos - retiros).
        curr = self.get_current_value(obj)
        return float(round(curr - self.get_initial_value(obj) - self.get_net_contributions(obj), 2))

    def get_total_return_pct(self, obj):
        # Rendimiento ponderado en el tiempo: no se ve afectado por depósitos ni retiros.
        from apps.accounting.services import equity_series, time_weighted_return
        return float(round(time_weighted_return(equity_series(obj)) * 100, 2))

    def get_primary_broker(self, obj):
        b = obj.broker_accounts.filter(trading_enabled=True).first()
        return BrokerAccountSerializer(b).data if b else None



class RecommendedAllocationSerializer(serializers.ModelSerializer):
    strategy_slug = serializers.CharField(source="strategy.slug", read_only=True)
    strategy_name = serializers.CharField(source="strategy.name", read_only=True)

    class Meta:
        model = RecommendedAllocation
        fields = ["id", "strategy", "strategy_slug", "strategy_name", "target_weight"]


class RecommendedPortfolioSerializer(serializers.ModelSerializer):
    allocations = RecommendedAllocationSerializer(many=True, read_only=True)

    class Meta:
        model = RecommendedPortfolio
        fields = ["id", "slug", "name", "description", "risk_level", "is_active", "allocations"]
