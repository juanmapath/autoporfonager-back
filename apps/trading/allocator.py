import uuid
from decimal import Decimal
from typing import List, Dict
from django.utils import timezone

from apps.portfolios.models import Portfolio, Allocation
from apps.trading.models import RebalanceRun, Target, OrderIntent
from apps.accounting.models import LogicalPosition
from apps.marketdata.models import LatestQuote
from apps.strategies.models import Signal
from apps.trading.rebalance_policy import is_periodic_rebalance_date, decide_position_trade
from core.market_time import NY_TZ
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
        processed_strategy_ids = set()

        # Rebalance policy (portfolio-level): entries / exits trade every day; resizing an open
        # position waits for a periodic rebalance date with drift above tolerance, unless the
        # allocation weights changed since the previous run or a manual rebalance was requested.
        is_rebalance_date = is_periodic_rebalance_date(portfolio, self.run.as_of.astimezone(NY_TZ).date())
        force_rebalance = portfolio.rebalance_pending
        previous_run = (
            RebalanceRun.objects.filter(portfolio=portfolio, status__in=["executing", "done"])
            .exclude(id=self.run.id).order_by("-as_of").first()
        )

        for alloc in allocations:
            strategy = alloc.strategy
            sleeve_capital = equity * alloc.target_weight
            version = alloc.version or strategy.versions.filter(status="live").first()
            if not version:
                continue
            processed_strategy_ids.add(strategy.id)
            allocation_changed = previous_run is not None and alloc.updated_at > previous_run.as_of

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

                # Smart accumulators add on every buy signal by design; everything else follows
                # the portfolio rebalance policy.
                if strat_kind == "smart_accumulator":
                    should_trade, reason = target_shares != curr_shares, "accumulate"
                else:
                    should_trade, reason = decide_position_trade(
                        portfolio, curr_shares, target_shares, price, equity,
                        is_rebalance_date=is_rebalance_date,
                        allocation_changed=allocation_changed,
                        force_rebalance=force_rebalance,
                    )
                delta = (target_shares - curr_shares) if should_trade else Decimal("0.0")

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
                    reason=reason,
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

        # --- LIQUIDATION SWEEP FOR STRATEGIES WITHOUT AN ACTIVE ALLOCATION ---
        # Positions held by strategies whose allocation was disabled / set to 0% (or removed) are not
        # visited by the loop above, so they would stay open forever. Liquidate them to 0 shares.
        stranded_positions = LogicalPosition.objects.filter(
            portfolio=portfolio,
            qty__gt=0,
        ).exclude(strategy_id__in=processed_strategy_ids).select_related("strategy", "instrument", "broker_account")

        for pos in stranded_positions:
            broker_acc = pos.broker_account or portfolio.broker_accounts.filter(trading_enabled=True).first()
            delta = -pos.qty

            t = Target.objects.create(
                run=self.run,
                strategy=pos.strategy,
                instrument=pos.instrument,
                broker_account=broker_acc,
                sleeve_capital=Decimal("0.00"),
                target_qty=Decimal("0.0"),
                current_qty=pos.qty,
                delta_qty=delta,
                reason="liquidate_disabled_allocation",
            )
            targets_created.append(t)

            ba_id = broker_acc.id if broker_acc else None
            key = (ba_id, pos.instrument_id)
            netting_map[key] = netting_map.get(key, Decimal("0.0")) + delta
            intent_inst_map[key] = (broker_acc, pos.instrument)

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
