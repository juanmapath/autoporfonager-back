import uuid
from decimal import Decimal
from typing import List, Dict
from django.utils import timezone

from apps.portfolios.models import Portfolio, Allocation
from apps.trading.models import RebalanceRun, Target, OrderIntent
from apps.accounting.models import LogicalPosition
from apps.marketdata.models import LatestQuote
from apps.strategies.models import Signal
from apps.trading.rebalance_policy import (
    is_portfolio_rebalance_date, is_bot_rebalance_date, bot_rebalance_config,
    needs_portfolio_reset, decide_position_trade,
)
from core.market_time import NY_TZ
from apps.trading.cash_sweep import (
    SWEEP_KIND, get_sweep_strategy, ensure_sweep_instrument, plan_sweep,
)
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
            from apps.accounting.services import portfolio_equity
            equity = portfolio_equity(portfolio)["equity"]

        # Contabilidad por manga: portafolios existentes se inicializan en su primer run (equity x peso).
        from apps.accounting.sleeves import (
            ensure_sleeve_accounts, release_disabled_sleeves, sleeve_cash, sleeve_values, add_entry,
        )
        ensure_sleeve_accounts(portfolio, equity=equity)

        allocations = list(portfolio.allocations.filter(enabled=True).select_related("strategy", "broker_account"))
        targets_created = []

        # Netting map: (broker_account_id, instrument_id) -> total_delta_qty
        netting_map: Dict[tuple, Decimal] = {}
        intent_inst_map = {}
        processed_strategy_ids = set()

        # Two-level rebalance policy.
        #  Level 1 (portfolio): each sleeve floats with its own PnL; on the portfolio rebalance date (and
        #    band exceeded), on a weight change or a manual request ALL sleeves are reset to
        #    equity x target_weight through cash transfers.
        #  Level 2 (bot): entries / exits trade every day; resizing an open position waits for the BOT's
        #    rebalance date with drift above the BOT's tolerance (inherited from the portfolio if unset).
        run_date = self.run.as_of.astimezone(NY_TZ).date()
        force_rebalance = portfolio.rebalance_pending
        previous_run = (
            RebalanceRun.objects.filter(portfolio=portfolio, status__in=["executing", "done"])
            .exclude(id=self.run.id).order_by("-as_of").first()
        )
        changed_ids = {
            a.strategy_id for a in allocations
            if previous_run is not None and a.updated_at > previous_run.as_of
        }

        release_disabled_sleeves(portfolio, reference=f"run:{self.run.id}")
        cash_by_sleeve = sleeve_cash(portfolio)
        values = {sid: v["value"] for sid, v in sleeve_values(portfolio).items()}
        weights = {a.strategy_id: a.target_weight for a in allocations}
        reset, reset_reason = needs_portfolio_reset(
            portfolio, values, weights, equity,
            is_rebalance_date=is_portfolio_rebalance_date(portfolio, run_date),
            allocation_changed=bool(changed_ids),
            force_rebalance=force_rebalance,
            missing_sleeve=any(sid not in cash_by_sleeve for sid in weights),
        )
        reset_sleeves = set()
        if reset:
            for a in allocations:
                transfer = equity * a.target_weight - values.get(a.strategy_id, Decimal("0"))
                if abs(transfer) >= Decimal("0.01"):
                    add_entry(portfolio, a.strategy_id, "transfer", transfer,
                              reference=f"run:{self.run.id}:{reset_reason}")
                    reset_sleeves.add(a.strategy_id)
            values = {sid: v["value"] for sid, v in sleeve_values(portfolio).items()}

        self.run.decisions = {
            **(self.run.decisions or {}),
            "portfolio_reset": reset_reason if reset else None,
            "sleeves": {
                str(a.strategy_id): {
                    "value": float(values.get(a.strategy_id, Decimal("0"))),
                    "weight_target": float(a.target_weight),
                    "weight_actual": float(values.get(a.strategy_id, Decimal("0")) / equity) if equity > 0 else 0.0,
                }
                for a in allocations
            },
            "unassigned": float(equity - sum((values.get(a.strategy_id, Decimal("0")) for a in allocations), Decimal("0"))),
        }
        self.run.save(update_fields=["decisions"])

        for alloc in allocations:
            strategy = alloc.strategy
            # The sleeve keeps its own PnL between portfolio rebalances (never negative for sizing).
            sleeve_capital = max(values.get(strategy.id, Decimal("0")), Decimal("0"))
            version = alloc.version or strategy.versions.filter(status="live").first()
            if not version:
                continue
            processed_strategy_ids.add(strategy.id)
            allocation_changed = strategy.id in changed_ids
            bot_config = bot_rebalance_config(version, portfolio)
            is_rebalance_date = is_bot_rebalance_date(version, portfolio, run_date)
            sleeve_was_reset = strategy.id in reset_sleeves

            # Fetch latest signals for instruments in this version
            strat_instruments = version.instruments.filter(role="traded").select_related("instrument")
            for si in strat_instruments:
                inst = si.instrument
                sig = Signal.objects.filter(version=version, instrument=inst).order_by("-as_of").first()
                exposure = sig.target_exposure if sig else Decimal("0.0")
                leverage_change = bool(sig and sig.changed and (sig.diagnostics or {}).get("leverage_resized"))

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
                        config=bot_config,
                        portfolio_reset=sleeve_was_reset,
                        leverage_change=leverage_change,
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
        ).exclude(strategy_id__in=processed_strategy_ids).exclude(strategy__kind=SWEEP_KIND).select_related(
            "strategy", "instrument", "broker_account",
        )

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

        # Cash sweep: free cash goes to T-bills (computed after every bot target is known).
        targets_created.extend(self._plan_cash_sweep(equity, targets_created, netting_map, intent_inst_map))

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

    def _plan_cash_sweep(self, equity, bot_targets, netting_map, intent_inst_map) -> List[Target]:
        """
        Objetivo del cash sweep (ver cash_sweep.py). Se calcula con la posicion final de TODOS los bots
        (cantidad actual + delta) para que el efectivo libre nunca compita con una compra de un bot.
        Con el sweep desactivado solo se liquida lo que quede de el.
        """
        portfolio = self.portfolio
        sweep_positions = list(
            LogicalPosition.objects.filter(portfolio=portfolio, strategy__kind=SWEEP_KIND, qty__gt=0)
            .select_related("strategy", "instrument", "broker_account")
        )
        if not portfolio.cash_sweep_enabled and not sweep_positions:
            return []

        strategy = get_sweep_strategy()
        broker_acc = portfolio.broker_accounts.filter(trading_enabled=True).first()
        price_cache: Dict[int, Decimal] = {}

        def price_of(inst):
            if inst.id not in price_cache:
                lq = LatestQuote.objects.filter(instrument=inst).first()
                price_cache[inst.id] = lq.price if lq else Decimal("100.00")
            return price_cache[inst.id]

        long_mv = Decimal("0")
        bot_buys = Decimal("0")
        for t in bot_targets:
            price = price_of(t.instrument)
            long_mv += (t.current_qty + t.delta_qty) * price
            if t.delta_qty > 0:
                bot_buys += t.delta_qty * price

        sweep_symbol = portfolio.cash_sweep_symbol
        created: List[Target] = []

        def add_target(inst, plan, current_qty, acc):
            t = Target.objects.create(
                run=self.run, strategy=strategy, instrument=inst, broker_account=acc,
                sleeve_capital=plan.target_value, target_qty=plan.target_qty, current_qty=current_qty,
                delta_qty=plan.delta_qty, reason=plan.reason,
            )
            created.append(t)
            key = (acc.id if acc else None, inst.id)
            netting_map[key] = netting_map.get(key, Decimal("0.0")) + plan.delta_qty
            intent_inst_map[key] = (acc, inst)

        # Positions in an ETF that is no longer the configured sweep symbol are liquidated.
        for pos in sweep_positions:
            if pos.instrument.symbol != sweep_symbol or not portfolio.cash_sweep_enabled:
                plan = plan_sweep(
                    enabled=False, equity=equity, long_mv=long_mv, buffer_pct=Decimal("0"),
                    min_trade_usd=Decimal("0"), price=price_of(pos.instrument), actual_qty=pos.qty,
                    bot_buys_value=bot_buys,
                )
                add_target(pos.instrument, plan, pos.qty, pos.broker_account or broker_acc)

        if portfolio.cash_sweep_enabled:
            inst = ensure_sweep_instrument(sweep_symbol)
            acc = broker_acc
            pos = LogicalPosition.objects.filter(
                portfolio=portfolio, strategy=strategy, instrument=inst, broker_account=acc,
            ).first()
            actual = pos.qty if pos else Decimal("0")
            plan = plan_sweep(
                enabled=True, equity=equity, long_mv=long_mv,
                buffer_pct=Decimal(str(portfolio.cash_sweep_buffer_pct)),
                min_trade_usd=Decimal(str(portfolio.cash_sweep_min_trade_usd)),
                price=price_of(inst), actual_qty=actual, bot_buys_value=bot_buys,
            )
            add_target(inst, plan, actual, acc)
        return created
