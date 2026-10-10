from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions, status
from decimal import Decimal
import pandas as pd

from apps.portfolios.models import Portfolio, RecommendedPortfolio, RecommendedAllocation
from apps.portfolios.serializers import RecommendedPortfolioSerializer
from apps.strategies.models import Strategy, StrategyVersion, BacktestRun
from apps.strategies.serializers import StrategySerializer, BacktestRunSerializer
from apps.strategies.backtest import BacktestEngine
from apps.risk.models import KillSwitch, FundingAlert
from apps.accounting.models import EquitySnapshot


class OpsOverviewView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        total_portfolios = Portfolio.objects.count()
        live_portfolios = Portfolio.objects.filter(kind="live", status="active").count()
        paper_portfolios = Portfolio.objects.filter(kind="paper", status="active").count()
        active_strategies = Strategy.objects.filter(is_active=True).count()
        open_alerts = FundingAlert.objects.filter(status="open").count()
        active_kill_switches = KillSwitch.objects.filter(is_active=True).count()

        # Approximate total AUM across active portfolios portably (SQLite & PostgreSQL compatible)
        total_aum = Decimal("0.00")
        for p in Portfolio.objects.filter(status="active"):
            last_snap = EquitySnapshot.objects.filter(portfolio=p).order_by("-date", "-created_at").first()
            if last_snap:
                total_aum += last_snap.total_equity
            else:
                total_aum += p.paper_initial_capital if p.kind == "paper" else Decimal("100000.00")

        if total_aum == Decimal("0.00"):
            total_aum = Decimal("100000.00")

        return Response({
            "aum_total": str(total_aum),
            "total_portfolios": total_portfolios,
            "live_portfolios": live_portfolios,
            "paper_portfolios": paper_portfolios,
            "active_strategies": active_strategies,
            "open_alerts": open_alerts,
            "active_kill_switches": active_kill_switches,
            "system_health": "NOMINAL" if active_kill_switches == 0 else "WARNING",
        })


class OpsStrategiesView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        from apps.strategies.signals.metadata import BOT_CATALOG, STRATEGIES_CATALOG
        from apps.fundamentals.metrics import get_metrics_catalog
        strategies = Strategy.objects.all().prefetch_related("versions")
        return Response({
            "strategies": StrategySerializer(strategies, many=True).data,
            "bot_types": BOT_CATALOG,
            "strategies_catalog": STRATEGIES_CATALOG,
            "metrics_catalog": get_metrics_catalog(),
        })

    def post(self, request):
        from apps.strategies.serializers import BotCreateSerializer
        serializer = BotCreateSerializer(data=request.data)
        if serializer.is_valid():
            bot_strategy = serializer.save()
            return Response(StrategySerializer(bot_strategy).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request):
        """Reconfigures an existing bot, creating a new StrategyVersion (e.g. v2, v3)."""
        from apps.strategies.serializers import BotCreateSerializer
        strategy_id = request.data.get("id") or request.data.get("strategy_id")
        strategy = Strategy.objects.filter(id=strategy_id).first()
        if not strategy:
            return Response({"error": "Bot / Strategy not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = BotCreateSerializer(data=request.data)
        if serializer.is_valid():
            updated = serializer.reconfigure(strategy, serializer.validated_data)
            return Response(StrategySerializer(updated).data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def patch(self, request):
        strategy_id = request.data.get("id")
        strategy = Strategy.objects.filter(id=strategy_id).first()
        if not strategy:
            return Response({"error": "Strategy not found"}, status=404)

        if "is_active" in request.data:
            strategy.is_active = bool(request.data["is_active"])
        if "family" in request.data:
            strategy.family = str(request.data["family"])
        if "decision_offset_minutes" in request.data:
            strategy.decision_offset_minutes = int(request.data["decision_offset_minutes"])

        strategy.save()
        return Response(StrategySerializer(strategy).data)

    def delete(self, request):
        """Deletes a bot/strategy ONLY if it is not allocated in any portfolio."""
        from apps.portfolios.models import Allocation
        strategy_id = request.data.get("id") or request.query_params.get("id")
        if not strategy_id:
            return Response({"error": "Parámetro 'id' es requerido para eliminar."}, status=status.HTTP_400_BAD_REQUEST)

        strategy = Strategy.objects.filter(id=strategy_id).first()
        if not strategy:
            return Response({"error": "Bot no encontrado."}, status=status.HTTP_404_NOT_FOUND)

        # Check if allocated in any active portfolio with target_weight > 0
        active_allocs = Allocation.objects.filter(strategy=strategy, target_weight__gt=0).select_related("portfolio")
        if active_allocs.exists():
            port_names = sorted(list(set(a.portfolio.name for a in active_allocs)))
            return Response({
                "error": f"No se puede eliminar el bot '{strategy.name}' porque está asignado activamente en los siguientes portafolios: {', '.join(port_names)}. Modifica la asignación en esos portafolios a 0% antes de eliminarlo."
            }, status=status.HTTP_400_BAD_REQUEST)

        # Remove zero-weight allocation records if any exist
        Allocation.objects.filter(strategy=strategy).delete()

        strat_name = strategy.name
        strategy.delete()

        return Response({
            "success": True,
            "message": f"Bot '{strat_name}' eliminado exitosamente."
        }, status=status.HTTP_200_OK)


class OpsRecommendedPortfoliosView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        recs = RecommendedPortfolio.objects.all().prefetch_related("allocations")
        return Response(RecommendedPortfolioSerializer(recs, many=True).data)

    def post(self, request):
        serializer = RecommendedPortfolioSerializer(data=request.data)
        if serializer.is_valid():
            rec = serializer.save()
            return Response(RecommendedPortfolioSerializer(rec).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


from django.utils import timezone


class OpsBacktestView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        runs = BacktestRun.objects.all().order_by("-created_at")[:20]
        return Response(BacktestRunSerializer(runs, many=True).data)

    def post(self, request):
        strategy_id = request.data.get("strategy_id")
        strat_slug = request.data.get("strategy_slug")
        bot_type = request.data.get("bot_type")
        traded_symbol = request.data.get("traded_symbol")
        signal_symbol = request.data.get("signal_symbol")
        strategies_config = request.data.get("strategies")

        strat = None
        if strategy_id:
            strat = Strategy.objects.filter(id=strategy_id).first()
        elif strat_slug:
            strat = Strategy.objects.filter(slug=strat_slug).first()

        today_str = timezone.now().strftime("%Y-%m-%d")
        default_start = (timezone.now() - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
        start_date = str(request.data.get("start_date") or default_start)
        end_date = str(request.data.get("end_date") or today_str)
        initial_capital = float(request.data.get("initial_capital", 100000.0))
        periodic_deposit = float(request.data.get("periodic_deposit", 0.0) or 0.0)
        deposit_frequency = str(request.data.get("deposit_frequency", "monthly") or "monthly")
        base_leverage = float(request.data.get("leverage") or request.data.get("base_leverage") or 1.0)
        max_leverage = float(request.data.get("max_leverage") or base_leverage)
        use_dynamic_leverage = bool(request.data.get("use_dynamic_leverage", request.data.get("use_regimes", False)))

        v = (strat.versions.filter(status="live").first() or strat.versions.order_by("-version").first()) if strat else None

        effective_traded_sym = (traded_symbol or "").strip().upper()
        if not effective_traded_sym and v:
            traded_inst = v.instruments.filter(role="traded").first()
            effective_traded_sym = traded_inst.instrument.symbol if traded_inst else "QQQ"
        if not effective_traded_sym:
            effective_traded_sym = "QQQ"

        slug_label = strat.slug if strat else f"adhoc-{effective_traded_sym.lower()}"

        run = BacktestRun.objects.create(
            version=v,
            strategy_slug=slug_label,
            instruments=[effective_traded_sym],
            start_date=start_date,
            end_date=end_date,
            initial_capital=Decimal(str(initial_capital)),
            status="running",
        )

        try:
            engine = BacktestEngine(initial_cash=initial_capital)

            if strategies_config or traded_symbol or not strat:
                res = engine.run_for_config(
                    bot_type=bot_type or (strat.kind if strat else "one_strategy"),
                    traded_symbol=effective_traded_sym,
                    signal_symbol=signal_symbol,
                    strategies=strategies_config or [],
                    start_date=start_date,
                    end_date=end_date,
                    leverage=base_leverage,
                    max_leverage=max_leverage,
                    use_dynamic_leverage=use_dynamic_leverage,
                    periodic_deposit=periodic_deposit,
                    deposit_frequency=deposit_frequency,
                )
            else:
                res = engine.run_for_strategy(
                    strat,
                    start_date=start_date,
                    end_date=end_date,
                    leverage=base_leverage,
                    max_leverage=max_leverage,
                    use_dynamic_leverage=use_dynamic_leverage,
                    periodic_deposit=periodic_deposit,
                    deposit_frequency=deposit_frequency,
                )

            run.status = "done"
            run.metrics = {
                "roi_pct": res["roi_pct"],
                "cagr_pct": res["cagr_pct"],
                "max_drawdown_pct": res["max_drawdown_pct"],
                "sharpe_ratio": res["sharpe_ratio"],
                "sortino_ratio": res["sortino_ratio"],
                "win_rate_pct": res["win_rate_pct"],
                "profit_factor": res["profit_factor"],
                "total_trades": res["total_trades"],
                "final_value": res["final_value"],
                "benchmark_final_value": res["benchmark_final_value"],
                "benchmark_roi_pct": res["benchmark_roi_pct"],
                "benchmark_cagr_pct": res["benchmark_cagr_pct"],
                "benchmark_max_drawdown_pct": res["benchmark_max_drawdown_pct"],
                "alpha_vs_benchmark": res["alpha_vs_benchmark"],
                "total_invested": res.get("total_invested", initial_capital),
                "periodic_deposit": res.get("periodic_deposit", 0.0),
                "deposit_frequency": res.get("deposit_frequency", "monthly"),
                "deposits_count": res.get("deposits_count", 0),
                "accumulated_shares": res.get("accumulated_shares", 0.0),
                "cash_remaining": res.get("cash_remaining", 0.0),
                "trades": res.get("trades", []),
                "trade_markers": res.get("trade_markers", []),
                "applied_leverage": res.get("applied_leverage", {}),
            }
            run.equity_curve = res["equity_curve"]
            run.save()

            data = BacktestRunSerializer(run).data
            data["trades"] = res.get("trades", [])
            data["trade_markers"] = res.get("trade_markers", [])
            data["applied_leverage"] = res.get("applied_leverage", {})
            data["symbol"] = res.get("symbol", effective_traded_sym)
            data["bot_type"] = res.get("bot_type", bot_type)
            data["total_invested"] = res.get("total_invested", initial_capital)
            data["periodic_deposit"] = res.get("periodic_deposit", 0.0)
            data["deposit_frequency"] = res.get("deposit_frequency", "monthly")
            data["accumulated_shares"] = res.get("accumulated_shares", 0.0)
            data["cash_remaining"] = res.get("cash_remaining", 0.0)
            return Response(data, status=status.HTTP_201_CREATED)
        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            run.save()
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class OpsKillSwitchView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def post(self, request):
        reason = request.data.get("reason", "Operator emergency trigger")
        ks = KillSwitch.objects.create(
            reason=reason,
            is_active=True,
        )
        # Immediately disable trading on all portfolios if system-wide
        Portfolio.objects.all().update(trading_enabled=False)
        return Response({
            "message": "Emergency Kill Switch activated. All trading disabled.",
            "kill_switch_id": ks.id,
            "reason": reason,
        })
