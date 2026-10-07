import logging
from decimal import Decimal
from django.utils import timezone
from apps.trading.models import OrderIntent, Order, Fill, FillAllocation
from apps.accounting.models import LogicalPosition, LedgerEntry
from apps.marketdata.models import LatestQuote
from apps.notifications.telegram import send_to_telegram, format_manual_orders_message
from core.market_time import get_decision_window, get_ny_now

logger = logging.getLogger(__name__)


class OrderDispatcher:
    """
    Executes netted OrderIntents across AUTO (Alpaca MOC), MANUAL (Yahoo Price + Telegram),
    and PAPER (App simulated close).
    """

    def dispatch_intent(self, intent: OrderIntent) -> Order:
        portfolio = intent.run.portfolio
        broker_account = intent.broker_account
        side = "buy" if intent.net_delta_qty > 0 else "sell"
        abs_qty = abs(intent.net_delta_qty)

        # 1. Check if execution is AUTO (Alpaca)
        if broker_account and broker_account.execution_mode == "auto":
            return self._dispatch_alpaca_moc(intent, side, abs_qty)

        # 2. Check if execution is MANUAL (Broker without API, e.g. eToro)
        elif broker_account and broker_account.execution_mode == "manual":
            return self._dispatch_manual(intent, side, abs_qty)

        # 3. Otherwise, PAPER Portfolio
        else:
            return self._dispatch_paper(intent, side, abs_qty)

    def _dispatch_alpaca_moc(self, intent: OrderIntent, side: str, qty: Decimal) -> Order:
        """Sends MOC order to Alpaca API before cutoff deadline."""
        now_ny = get_ny_now()
        window = get_decision_window(now_ny.date())
        if window:
            _, cutoff = window
            if now_ny > cutoff:
                intent.status = "cancelled"
                intent.save(update_fields=["status"])
                intent.run.status = "missed_cutoff"
                intent.run.save(update_fields=["status"])
                send_to_telegram(f"🚨 [MISSED CUTOFF] Orden cancelada para {intent.instrument.symbol}. Superado corte MOC.")
                raise RuntimeError("MOC deadline cutoff passed. Order rejected for safety.")

        order = Order.objects.create(
            intent=intent,
            portfolio=intent.run.portfolio,
            broker_account=intent.broker_account,
            instrument=intent.instrument,
            client_order_id=intent.idempotency_key,
            source="broker",
            order_type="moc",
            side=side,
            qty=qty,
            status="submitted",
        )

        try:
            # Connect via alpaca-py TradingClient
            creds = intent.broker_account.credential.get_secrets() if intent.broker_account.credential else {}
            api_key = creds.get("key_id")
            api_secret = creds.get("secret")
            is_paper = intent.broker_account.environment == "paper"

            if api_key and api_secret:
                from alpaca.trading.client import TradingClient
                from alpaca.trading.requests import MarketOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce

                client = TradingClient(api_key, api_secret, paper=is_paper)
                req = MarketOrderRequest(
                    symbol=intent.instrument.symbol,
                    qty=float(qty),
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.CLS,
                    client_order_id=intent.idempotency_key,
                )
                alpaca_res = client.submit_order(order_data=req)
                order.broker_order_id = str(alpaca_res.id)
                order.status = "accepted"
                order.save(update_fields=["broker_order_id", "status"])
            else:
                logger.warning(f"No Alpaca API credentials configured for {intent.broker_account}. Marked accepted for mock.")
                order.status = "accepted"
                order.save(update_fields=["status"])

        except Exception as e:
            logger.error(f"Alpaca MOC dispatch failed for {intent.idempotency_key}: {e}")
            order.status = "rejected"
            order.save(update_fields=["status"])

        intent.status = "dispatched"
        intent.save(update_fields=["status"])
        return order

    def _dispatch_manual(self, intent: OrderIntent, side: str, qty: Decimal) -> Order:
        """Executes simulated manual order at LatestQuote price and fires Telegram alert."""
        lq = LatestQuote.objects.filter(instrument=intent.instrument).first()
        fill_price = lq.price if lq else Decimal("100.00")

        order = Order.objects.create(
            intent=intent,
            portfolio=intent.run.portfolio,
            broker_account=intent.broker_account,
            instrument=intent.instrument,
            client_order_id=intent.idempotency_key,
            source="simulated_manual",
            order_type="moc",
            side=side,
            qty=qty,
            status="filled",
        )

        fill = Fill.objects.create(
            order=order,
            fill_price=fill_price,
            qty=qty,
            commission=Decimal("1.00"),
            source="simulated_manual",
            filled_at=timezone.now(),
        )

        # Update logical positions and bookkeeping
        self._record_fill_bookkeeping(fill, intent.run.portfolio, intent.broker_account, intent.instrument, side, qty, fill_price)

        # Send Telegram notification
        msg = format_manual_orders_message(
            portfolio_name=intent.run.portfolio.name,
            broker_name=intent.broker_account.display_name if intent.broker_account else "Manual",
            rebalance_reason=intent.run.trigger,
            orders=[{
                "side": side,
                "qty": float(qty),
                "symbol": intent.instrument.symbol,
                "price": float(fill_price),
            }]
        )
        send_to_telegram(msg)

        intent.status = "dispatched"
        intent.save(update_fields=["status"])
        return order

    def _dispatch_paper(self, intent: OrderIntent, side: str, qty: Decimal) -> Order:
        """Paper simulation at decision window price."""
        lq = LatestQuote.objects.filter(instrument=intent.instrument).first()
        fill_price = lq.price if lq else Decimal("100.00")

        order = Order.objects.create(
            intent=intent,
            portfolio=intent.run.portfolio,
            instrument=intent.instrument,
            client_order_id=intent.idempotency_key,
            source="simulated_paper",
            order_type="moc",
            side=side,
            qty=qty,
            status="filled",
        )

        fill = Fill.objects.create(
            order=order,
            fill_price=fill_price,
            qty=qty,
            commission=Decimal("0.00"),
            source="simulated_paper",
            filled_at=timezone.now(),
        )

        self._record_fill_bookkeeping(fill, intent.run.portfolio, None, intent.instrument, side, qty, fill_price)

        intent.status = "dispatched"
        intent.save(update_fields=["status"])
        return order

    def _record_fill_bookkeeping(self, fill, portfolio, broker_account, instrument, side, qty, price):
        # Attribute to primary allocation
        alloc = portfolio.allocations.filter(enabled=True).first()
        strat = alloc.strategy if alloc else None
        if strat:
            FillAllocation.objects.create(
                fill=fill,
                strategy=strat,
                allocated_qty=qty,
            )
            # Update or create LogicalPosition
            pos, created = LogicalPosition.objects.get_or_create(
                portfolio=portfolio,
                broker_account=broker_account,
                strategy=strat,
                instrument=instrument,
                defaults={"qty": Decimal("0.0"), "avg_cost": price},
            )
            if side == "buy":
                new_qty = pos.qty + qty
                if new_qty > 0:
                    pos.avg_cost = ((pos.qty * pos.avg_cost) + (qty * price)) / new_qty
                pos.qty = new_qty
            else:
                pos.qty = max(Decimal("0.0"), pos.qty - qty)
            pos.save()
