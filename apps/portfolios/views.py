from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from decimal import Decimal

from apps.portfolios.models import Portfolio, Allocation, AllocationChange
from apps.portfolios.serializers import PortfolioSerializer, AllocationSerializer
from apps.brokers.models import BrokerAccount
from apps.brokers.serializers import BrokerAccountSerializer
from apps.accounting.models import LogicalPosition
from apps.trading.models import Order
from apps.strategies.models import Strategy
from apps.strategies.serializers import StrategySerializer


def _jsonable(row: dict) -> dict:
    """Convierte Decimals a float redondeado para la respuesta JSON."""
    return {k: (float(round(v, 6)) if isinstance(v, Decimal) else v) for k, v in row.items()}


class MeView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user
        plan = user.effective_plan
        live_count = Portfolio.objects.filter(user=user, kind="live").exclude(status="closed").count()
        paper_count = Portfolio.objects.filter(user=user, kind="paper").exclude(status="closed").count()

        return Response({
            "email": user.email,
            "username": user.username,
            "timezone": user.timezone,
            "is_staff": user.is_staff,
            "plan": {
                "name": plan.name,
                "code": plan.code,
                "max_live_portfolios": plan.max_live_portfolios,
                "max_paper_portfolios": plan.max_paper_portfolios,
                "max_broker_accounts_per_portfolio": plan.max_broker_accounts_per_portfolio,
            },
            "usage": {
                "live_portfolios": f"{live_count}/{plan.max_live_portfolios}",
                "paper_portfolios": f"{paper_count}/{plan.max_paper_portfolios}",
            }
        })


class PortfolioViewSet(viewsets.ModelViewSet):
    serializer_class = PortfolioSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Strict multi-tenant isolation: User only sees their own portfolios
        return Portfolio.objects.filter(user=self.request.user).order_by("-created_at")

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=["get", "put"])
    def allocations(self, request, pk=None):
        portfolio = self.get_object()
        if request.method == "GET":
            allocs = portfolio.allocations.all()
            return Response(AllocationSerializer(allocs, many=True).data)

        # PUT: Update allocations
        alloc_data = request.data.get("allocations", [])
        total_weight = sum(Decimal(str(item.get("target_weight", 0))) for item in alloc_data)
        if total_weight > Decimal("1.0001"):
            return Response({"error": f"Total weight cannot exceed 100% (got {total_weight * 100}%)"}, status=400)

        active_broker = portfolio.broker_accounts.filter(trading_enabled=True).first()
        saved = []
        for item in alloc_data:
            strat_id = item.get("strategy")
            target_weight = Decimal(str(item.get("target_weight", 0)))
            broker_acc_id = item.get("broker_account") or (active_broker.id if active_broker else None)
            alloc, created = Allocation.objects.update_or_create(
                portfolio=portfolio,
                strategy_id=strat_id,
                defaults={
                    "target_weight": target_weight,
                    "enabled": target_weight > 0,
                    "broker_account_id": broker_acc_id,
                }
            )
            saved.append(alloc)

        return Response(AllocationSerializer(saved, many=True).data)

    @action(detail=True, methods=["get", "post"], url_path="broker-accounts")
    def broker_accounts(self, request, pk=None):
        portfolio = self.get_object()
        if request.method == "GET":
            accounts = portfolio.broker_accounts.all()
            return Response(BrokerAccountSerializer(accounts, many=True).data)

        # POST: Create and validate broker account
        from apps.brokers.serializers import BrokerAccountCreateSerializer
        serializer = BrokerAccountCreateSerializer(data=request.data)
        if serializer.is_valid():
            try:
                account = serializer.create_for_portfolio(portfolio, serializer.validated_data)
                return Response(BrokerAccountSerializer(account).data, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=["post"], url_path="test-broker-connection")
    def test_broker_connection(self, request):
        """Tests credentials against the provider API (e.g. Alpaca ping /v2/account) before saving."""
        from apps.brokers.serializers import BrokerTestConnectionSerializer
        serializer = BrokerTestConnectionSerializer(data=request.data)
        if serializer.is_valid():
            return Response({
                "success": True,
                "account": serializer.validated_data.get("alpaca_account", {}),
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=["post"], url_path="sync-broker")
    def sync_broker(self, request, pk=None):
        """Pings connected broker (e.g. Alpaca) to refresh latest equity, cash, and buying power."""
        portfolio = self.get_object()
        account = portfolio.broker_accounts.filter(trading_enabled=True).first()
        if not account or not account.credential:
            return Response({"error": "No hay cuenta de broker conectada a este portafolio."}, status=status.HTTP_400_BAD_REQUEST)

        from apps.brokers.services import sync_alpaca_account

        if account.provider == "alpaca":
            # El capital inicial se fija al vincular el broker; aquí sólo se refrescan saldo y movimientos.
            res = sync_alpaca_account(account)
            if not res.get("success"):
                return Response({"error": f"Error al sincronizar con Alpaca: {res.get('error')}"}, status=status.HTTP_400_BAD_REQUEST)

            return Response({
                "success": True,
                "account": BrokerAccountSerializer(account).data,
                "portfolio": PortfolioSerializer(portfolio).data,
            })
        return Response({"error": f"Proveedor '{account.provider}' no soportado para sincronización."}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=["post", "delete"], url_path="disconnect-broker")
    def disconnect_broker(self, request, pk=None):
        """Unlinks broker accounts and credentials from this portfolio."""
        portfolio = self.get_object()
        accounts = list(portfolio.broker_accounts.all())
        if not accounts:
            return Response({"message": "No hay cuentas vinculadas."}, status=status.HTTP_200_OK)

        # Remove allocation associations
        portfolio.allocations.update(broker_account=None)

        for acc in accounts:
            cred = acc.credential
            acc.delete()
            if cred:
                cred.delete()

        return Response({
            "success": True,
            "message": "Broker desvinculado exitosamente.",
            "portfolio": PortfolioSerializer(portfolio).data,
        })

    @action(detail=True, methods=["get"])
    def positions(self, request, pk=None):
        """Posiciones abiertas valoradas a precio actual, con PnL no realizado y apalancamiento."""
        from apps.accounting.services import open_positions
        portfolio = self.get_object()
        result = open_positions(portfolio, refresh=True)
        return Response({
            "positions": [_jsonable(r) for r in result["positions"]],
            "summary": _jsonable(result["summary"]),
        })

    @action(detail=True, methods=["get"], url_path="equity-history")
    def equity_history(self, request, pk=None):
        """Historial diario de equity y cierres de los índices de referencia (?benchmarks=SPY,QQQ)."""
        from apps.accounting.services import equity_history, BENCHMARKS
        portfolio = self.get_object()
        raw = request.query_params.get("benchmarks", "SPY,QQQ")
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        data = equity_history(portfolio, symbols)
        return Response({
            "portfolio": [_jsonable(p) for p in data["portfolio"]],
            "benchmarks": {
                sym: {"name": b["name"], "series": [_jsonable(p) for p in b["series"]]}
                for sym, b in data["benchmarks"].items()
            },
            "available_benchmarks": BENCHMARKS,
        })

    @action(detail=True, methods=["get"])
    def orders(self, request, pk=None):
        portfolio = self.get_object()
        orders = (
            Order.objects.filter(portfolio=portfolio)
            .select_related("instrument")
            .prefetch_related("fills")
            .order_by("-created_at")[:50]
        )
        data = []
        for o in orders:
            fills = list(o.fills.all())
            filled_qty = sum((f.qty for f in fills), Decimal("0"))
            avg_price = (sum((f.qty * f.fill_price for f in fills), Decimal("0")) / filled_qty) if filled_qty else None
            data.append({
                "id": o.id,
                "client_order_id": o.client_order_id,
                "instrument": o.instrument.symbol,
                "side": o.side,
                "qty": str(o.qty),
                "fill_price": str(round(avg_price, 4)) if avg_price is not None else None,
                "commission": str(sum((f.commission for f in fills), Decimal("0"))),
                "filled_at": max((f.filled_at for f in fills), default=None),
                "source": o.source,
                "status": o.status,
                "created_at": o.created_at,
            })
        return Response(data)

    @action(detail=True, methods=["post"], url_path="rebalance-preview")
    def rebalance_preview(self, request, pk=None):
        """Simulates rebalance calculation without placing any real or broker orders."""
        portfolio = self.get_object()
        allocs = portfolio.allocations.filter(enabled=True)
        preview_targets = []
        from apps.accounting.services import portfolio_equity
        equity = portfolio_equity(portfolio)["equity"]

        for a in allocs:
            sleeve_val = equity * a.target_weight
            preview_targets.append({
                "strategy": a.strategy.name,
                "target_weight": str(a.target_weight),
                "target_capital": str(sleeve_val),
            })
        return Response({
            "portfolio_id": portfolio.id,
            "estimated_equity": str(equity),
            "targets": preview_targets,
        })

    @action(detail=True, methods=["get"], url_path="drift")
    def drift(self, request, pk=None):
        """Calculates current drift and whether rebalancing threshold is exceeded."""
        portfolio = self.get_object()
        return Response(portfolio.calculate_drift_metrics())


class StrategyCatalogView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        from apps.strategies.signals.metadata import BOT_CATALOG, STRATEGIES_CATALOG
        strategies = Strategy.objects.filter(is_active=True).prefetch_related("versions")
        return Response({
            "instances": StrategySerializer(strategies, many=True).data,
            "bot_types": BOT_CATALOG,
            "strategies_catalog": STRATEGIES_CATALOG,
        })
