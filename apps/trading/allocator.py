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

                # Fallback to portfolio's active broker account if allocation doesn't specify one
                broker_acc = alloc.broker_account or portfolio.broker_accounts.filter(trading_enabled=True).first()

                # Current position
                log_pos = LogicalPosition.objects.filter(
                    portfolio=portfolio,
                    strategy=strategy,
                    instrument=inst,
                    broker_account=broker_acc,
                ).first()
                curr_shares = log_pos.qty if log_pos else Decimal("0.0")

                # Strategy-specific sizing rules:
                strat_kind = strategy.kind
                strat_params = version.params or {}

                # 1. HOLD BOT: Check if frozen (participates_in_rebalance=False)
                if strat_kind == "hold":
                    participates = strat_params.get("participates_in_rebalance", True)
                    is_cash = strat_params.get("is_cash", False) or inst.symbol.upper() in ("CASH", "USD")
                    if is_cash:
                        # Cash reserve: holds 0 shares in equity
                        target_shares = Decimal("0.0")
                    elif not participates and curr_shares > 0:
                        # Frozen position: lock shares, zero rebalance delta
                        target_shares = curr_shares

                # 2. SMART ACCUMULATOR BOT: Asymmetric accumulation (never sells on neutral/flat)
                elif strat_kind == "smart_accumulator":
                    if exposure > 0:
                        # Buying dip: use sleeve capital available to add to position
                        # Target shares expands to absorb allocated capital
                        target_shares = max(curr_shares, target_shares)
                    else:
                        # Neutral/flat signal: KEEP accumulated shares (never sell)
                        target_shares = curr_shares

                delta = target_shares - curr_shares

                # Create immutable Target
                t = Target.objects.create(
                    run=self.run,
                    strategy=strategy,
                    instrument=inst,
                    broker_account=broker_acc,
                    sleeve_capital=sleeve_capital,
                    target_qty=target_shares,
                    current_qty=curr_shares,
                    delta_qty=delta,
                    reason="signal_or_drift",
                )
                targets_created.append(t)

                # Add to netting
                ba_id = broker_acc.id if broker_acc else None
                key = (ba_id, inst.id)
                netting_map[key] = netting_map.get(key, Decimal("0.0")) + delta
                intent_inst_map[key] = (broker_acc, inst)

            # --- LIQUIDATION SWEEP FOR REMOVED ASSETS ---
            # If the strategy previously held positions in instruments that are NO LONGER in this version,
            # liquidate them to 0 shares (target_shares = 0, delta = -curr_shares)
            active_inst_ids = set(strat_instruments.values_list("instrument_id", flat=True))
            orphan_positions = LogicalPosition.objects.filter(
                portfolio=portfolio,
                strategy=strategy,
                qty__gt=0,
            ).exclude(instrument_id__in=active_inst_ids).select_related("instrument", "broker_account")

            for orph in orphan_positions:
                inst = orph.instrument
                broker_acc = orph.broker_account or alloc.broker_account or portfolio.broker_accounts.filter(trading_enabled=True).first()
                curr_shares = orph.qty
                target_shares = Decimal("0.0")
                delta = -curr_shares

                t = Target.objects.create(
                    run=self.run,
                    strategy=strategy,
                    instrument=inst,
                    broker_account=broker_acc,
                    sleeve_capital=sleeve_capital,
                    target_qty=target_shares,
                    current_qty=curr_shares,
                    delta_qty=delta,
                    reason="liquidate_removed_asset",
                )
                targets_created.append(t)

                ba_id = broker_acc.id if broker_acc else None
                key = (ba_id, inst.id)
                netting_map[key] = netting_map.get(key, Decimal("0.0")) + delta
                intent_inst_map[key] = (broker_acc, inst)

        # Generate netted OrderIntents (with rebalance band threshold: 2.5% drift or >= $25 order)
        REBALANCE_BAND_PP = Decimal("0.025")
        MIN_ORDER_USD = Decimal("25.00")

        intents = []
        for (ba_id, inst_id), net_delta in netting_map.items():
            if abs(net_delta) < Decimal("0.0001"):
                continue  # zero net delta, no trade needed

            broker_account, instrument = intent_inst_map[(ba_id, inst_id)]
            lq = LatestQuote.objects.filter(instrument=instrument).first()
            p = lq.price if lq else Decimal("100.00")
            trade_value = abs(net_delta * p)

            # Skip small churn orders unless position is being completely liquidated or initialized
            # (trade_value < $25 and drift within rebalance band)
            if equity > 0 and (trade_value / equity) < REBALANCE_BAND_PP and trade_value < MIN_ORDER_USD:
                continue

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
