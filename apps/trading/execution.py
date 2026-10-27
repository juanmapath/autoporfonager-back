import logging
from decimal import Decimal
from django.utils import timezone
from apps.trading.models import OrderIntent, Order, Fill, FillAllocation
from apps.accounting.models import LogicalPosition, LedgerEntry
from apps.marketdata.models import LatestQuote
from apps.notifications.telegram import send_to_telegram, format_manual_orders_message
from core.market_time import get_decision_window, get_ny_now
from core.money import quantize_qty, quantize_amount

logger = logging.getLogger(__name__)


class OrderDispatcher:
    """
    Executes netted OrderIntents across AUTO (Alpaca DAY market), MANUAL (Yahoo Price + Telegram),
    and PAPER (App simulated close).
    """

    def dispatch_intent(self, intent: OrderIntent) -> Order:
        existing_order = Order.objects.filter(intent=intent).first()
        if existing_order:
            if intent.status != "dispatched":
                intent.status = "dispatched"
                intent.save(update_fields=["status"])
            return existing_order

        portfolio = intent.run.portfolio
        broker_account = intent.broker_account
        side = "buy" if intent.net_delta_qty > 0 else "sell"
        abs_qty = abs(intent.net_delta_qty)

        # 1. Check if execution is AUTO (Alpaca)
        if broker_account and broker_account.execution_mode == "auto":
            return self._dispatch_alpaca_market(intent, side, abs_qty)

        # 2. Check if execution is MANUAL (Broker without API, e.g. eToro)
        elif broker_account and broker_account.execution_mode == "manual":
            return self._dispatch_manual(intent, side, abs_qty)

        # 3. Otherwise, PAPER Portfolio
        else:
            return self._dispatch_paper(intent, side, abs_qty)

    def _dispatch_alpaca_market(self, intent: OrderIntent, side: str, qty: Decimal) -> Order:
        """
        Sends a DAY market order to Alpaca just before the close.
        MOC (TimeInForce.CLS) is not used: Alpaca rejects it for fractional quantities.
        """
        now_ny = get_ny_now()
        window = get_decision_window(now_ny.date())
        # Sin ventana (feriado o fin de semana) se rechaza igual que tras el corte: una orden DAY
        # quedaría encolada en Alpaca para la apertura de la siguiente sesión.
        if window is None or now_ny > window[1]:
            reason = "Superado corte de envío antes del cierre." if window else "Hoy no hay sesión NYSE (sin ventana de decisión)."
            intent.status = "cancelled"
            intent.save(update_fields=["status"])
            intent.run.status = "missed_cutoff"
            intent.run.save(update_fields=["status"])
            send_to_telegram(f"🚨 [MISSED CUTOFF] Orden cancelada para {intent.instrument.symbol}. {reason}")
            raise RuntimeError(f"Order rejected for safety: {reason}")

        order = Order.objects.create(
            intent=intent,
            portfolio=intent.run.portfolio,
            broker_account=intent.broker_account,
            instrument=intent.instrument,
            client_order_id=intent.idempotency_key,
            source="broker",
            order_type="market",
            side=side,
            qty=qty,
            status="submitted",
        )

        try:
            # Connect via alpaca-py TradingClient
            creds = intent.broker_account.credential.get_secrets() if intent.broker_account.credential else {}
            api_key = creds.get("key_id") or creds.get("api_key")
            api_secret = creds.get("secret_key") or creds.get("secret")
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
                    time_in_force=TimeInForce.DAY,
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
            logger.error(f"Alpaca order dispatch failed for {intent.idempotency_key}: {e}")
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
        self._record_fill_bookkeeping(fill, intent, intent.instrument, side, qty, fill_price)

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

        self._record_fill_bookkeeping(fill, intent, intent.instrument, side, qty, fill_price)

        intent.status = "dispatched"
        intent.save(update_fields=["status"])
        return order

    def _record_fill_bookkeeping(self, fill, intent, instrument, side, qty, price):
        """
        Atribuye el fill a las estrategias que originaron la orden, usando los Target del run.
        La orden es el neto de los delta_qty de cada estrategia para ese instrumento, así que cada
        estrategia recibe su propio delta (escalado si el fill fue parcial), incluidos los cruces
        internos entre estrategias que el netting evitó enviar al broker.
        """
        if intent is None:
            logger.error(f"Fill {fill.id} sin OrderIntent asociado: no se puede atribuir a una estrategia.")
            return

        targets = [
            t for t in intent.run.targets.filter(
                instrument=instrument, broker_account=intent.broker_account,
            ).select_related("strategy", "broker_account")
            if t.delta_qty != 0
        ]
        net_delta = sum((t.delta_qty for t in targets), Decimal("0"))
        if not targets or net_delta == 0:
            logger.error(f"Fill {fill.id} ({instrument.symbol}) sin targets con delta en el run {intent.run_id}: no atribuido.")
            return

        signed_fill = qty if side == "buy" else -qty
        fill_ratio = signed_fill / net_delta
        portfolio = intent.run.portfolio

        for target in targets:
            allocated = quantize_qty(target.delta_qty * fill_ratio, round_down=False)
            if allocated == 0:
                continue

            FillAllocation.objects.create(
                fill=fill,
                strategy=target.strategy,
                allocated_qty=allocated,
            )
            pos, _ = LogicalPosition.objects.get_or_create(
                portfolio=portfolio,
                broker_account=target.broker_account,
                strategy=target.strategy,
                instrument=instrument,
                defaults={"qty": Decimal("0.0"), "avg_cost": price},
            )
            if allocated > 0:
                new_qty = pos.qty + allocated
                pos.avg_cost = ((pos.qty * pos.avg_cost) + (allocated * price)) / new_qty
                pos.qty = new_qty
            else:
                sold = min(pos.qty, -allocated)
                pos.realized_pnl += quantize_amount((price - pos.avg_cost) * sold)
                pos.qty -= sold
            pos.save()
