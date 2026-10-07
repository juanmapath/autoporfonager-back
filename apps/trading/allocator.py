import uuid
from decimal import Decimal
from typing import List, Dict
from django.utils import timezone

from apps.portfolios.models import Portfolio, Allocation
from apps.trading.models import RebalanceRun, Target, OrderIntent
from apps.accounting.models import LogicalPosition
from apps.marketdata.models import LatestQuote
from apps.strategies.models import Signal
from core.money import quantize_qty



class PortfolioAllocator:
    """
    Sizing, Multi-Strategy Netting, and Order Intent Generator.
    Ensures that multiple strategy sleeves targeting the same instrument net out
    before dispatching to broker accounts, preventing unnecessary churn.
    """

    def __init__(self, run: RebalanceRun):
        self.run = run
        self.portfolio: Portfolio = run.portfolio

    def generate_targets_and_intents(self) -> List[OrderIntent]:
        portfolio = self.portfolio
        equity = self.run.equity_base
        if equity <= 0:
            equity = portfolio.paper_initial_capital if portfolio.kind == "paper" else Decimal("100000.00")

        allocations = portfolio.allocations.filter(enabled=True).select_related("strategy", "broker_account")
        targets_created = []

        # Netting map: (broker_account_id, instrument_id) -> total_delta_qty
        netting_map: Dict[tuple, Decimal] = {}
        intent_inst_map = {}

        for alloc in allocations:
            strategy = alloc.strategy
            sleeve_capital = equity * alloc.target_weight
            version = alloc.version or strategy.versions.filter(status="live").first()
            if not version:
                continue

            # Fetch latest signals for instruments in this version
            strat_instruments = version.instruments.filter(role="traded").select_related("instrument")
            for si in strat_instruments:
                inst = si.instrument
                sig = Signal.objects.filter(version=version, instrument=inst).order_by("-as_of").first()
                exposure = sig.target_exposure if sig else Decimal("0.0")

                # Fetch latest price
                lq = LatestQuote.objects.filter(instrument=inst).first()
                price = lq.price if lq else Decimal("100.00")

                # Sizing
                target_value = sleeve_capital * exposure * si.weight
                target_shares = target_value / price if price > 0 else Decimal("0.0")
                target_shares = quantize_qty(target_shares, round_down=True)

                # Current position
                log_pos = LogicalPosition.objects.filter(
                    portfolio=portfolio,
                    strategy=strategy,
                    instrument=inst,
                    broker_account=alloc.broker_account,
                ).first()
                curr_shares = log_pos.qty if log_pos else Decimal("0.0")

                delta = target_shares - curr_shares

                # Create immutable Target
                t = Target.objects.create(
                    run=self.run,
                    strategy=strategy,
                    instrument=inst,
                    broker_account=alloc.broker_account,
                    sleeve_capital=sleeve_capital,
                    target_qty=target_shares,
                    current_qty=curr_shares,
                    delta_qty=delta,
                    reason="signal_or_drift",
                )
                targets_created.append(t)

                # Add to netting
                key = (alloc.broker_account_id, inst.id)
                netting_map[key] = netting_map.get(key, Decimal("0.0")) + delta
                intent_inst_map[key] = (alloc.broker_account, inst)

        # Generate netted OrderIntents
        intents = []
        for (ba_id, inst_id), net_delta in netting_map.items():
            if abs(net_delta) < Decimal("0.0001"):
                continue  # zero net delta, no trade needed

            broker_account, instrument = intent_inst_map[(ba_id, inst_id)]
            idempotency_key = f"run_{self.run.id}_ba_{ba_id or 'paper'}_inst_{instrument.symbol}_{uuid.uuid4().hex[:6]}"

            intent = OrderIntent.objects.create(
                run=self.run,
                broker_account=broker_account,
                instrument=instrument,
                net_delta_qty=net_delta,
                idempotency_key=idempotency_key,
                status="pending",
            )
            intents.append(intent)

        return intents
