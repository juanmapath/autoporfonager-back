from rest_framework import serializers
from decimal import Decimal
from apps.portfolios.models import Portfolio, Allocation, RecommendedPortfolio, RecommendedAllocation
from apps.brokers.serializers import BrokerAccountSerializer
from apps.strategies.serializers import StrategySerializer


class AllocationSerializer(serializers.ModelSerializer):
    strategy_name = serializers.CharField(source="strategy.name", read_only=True)
    strategy_slug = serializers.CharField(source="strategy.slug", read_only=True)

    class Meta:
        model = Allocation
        fields = [
            "id", "portfolio", "strategy", "strategy_name", "strategy_slug",
            "target_weight", "broker_account", "enabled",
        ]
        read_only_fields = ["id"]


class PortfolioSerializer(serializers.ModelSerializer):
    allocations = AllocationSerializer(many=True, read_only=True)
    broker_accounts = BrokerAccountSerializer(many=True, read_only=True)
    current_value = serializers.SerializerMethodField()
    initial_value = serializers.SerializerMethodField()
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
            "current_value", "initial_value", "total_pnl", "total_return_pct", "primary_broker",
            "created_at",
        ]
        read_only_fields = ["id", "user", "created_at"]

    def get_current_value(self, obj):
        # 1. If has connected broker account with managed_capital, return it
        active_broker = obj.broker_accounts.filter(trading_enabled=True).first()
        if active_broker and active_broker.managed_capital:
            return float(active_broker.managed_capital)
        # 2. Else return paper capital
        return float(obj.paper_initial_capital if obj.kind == "paper" else Decimal("100000.00"))

    def get_initial_value(self, obj):
        return float(obj.paper_initial_capital if obj.kind == "paper" else Decimal("100000.00"))

    def get_total_pnl(self, obj):
        curr = self.get_current_value(obj)
        init = self.get_initial_value(obj)
        return float(round(curr - init, 2))

    def get_total_return_pct(self, obj):
        curr = self.get_current_value(obj)
        init = self.get_initial_value(obj)
        if init > 0:
            return float(round(((curr - init) / init) * 100, 2))
        return 0.0

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
