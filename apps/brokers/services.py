import logging
from datetime import date, timedelta
from decimal import Decimal

from apps.brokers.models import BrokerAccount
from core.market_time import NY_TZ
from apps.brokers.providers.alpaca import get_alpaca_cash_activities, test_alpaca_credentials

logger = logging.getLogger(__name__)


def _alpaca_keys(account: BrokerAccount):
    secrets = account.credential.get_secrets()
    return (
        secrets.get("key_id") or secrets.get("api_key"),
        secrets.get("secret_key") or secrets.get("secret"),
        secrets.get("environment", account.environment),
    )


def sync_alpaca_balance(account: BrokerAccount) -> dict:
    """
    Refresca equity, cash y buying power de una cuenta Alpaca en `managed_capital` / `order_policy`.
    Devuelve el resultado del ping a Alpaca (con `success`).
    """
    if account.provider != "alpaca" or not account.credential:
        return {"success": False, "error": "La cuenta no es Alpaca o no tiene credenciales."}

    res = test_alpaca_credentials(*_alpaca_keys(account))
    if not res.get("success"):
        logger.error(f"No se pudo sincronizar el saldo de {account.display_name}: {res.get('error')}")
        return res

    account.managed_capital = Decimal(str(res.get("portfolio_value") or res.get("cash")))
    account.order_policy = {
        "cash": res.get("cash"),
        "buying_power": res.get("buying_power"),
        "currency": res.get("currency", "USD"),
        "status": res.get("status"),
    }
    account.save(update_fields=["managed_capital", "order_policy", "updated_at"])
    return res


def sync_alpaca_cash_flows(account: BrokerAccount) -> int:
    """
    Importa al ledger los depósitos y retiros de la cuenta Alpaca posteriores a su vinculación
    (los anteriores ya forman parte del capital inicial). Idempotente por id de actividad.
    Devuelve el número de movimientos nuevos.
    """
    from apps.accounting.models import LedgerEntry

    if account.provider != "alpaca" or not account.credential:
        return 0

    flows = LedgerEntry.objects.filter(broker_account=account, entry_type__in=["deposit", "withdrawal"])
    last = flows.exclude(effective_date=None).order_by("-effective_date").values_list("effective_date", flat=True).first()
    # Alpaca fecha las actividades en hora de Nueva York
    linked_on = account.created_at.astimezone(NY_TZ).date()
    # Se relee un margen de días por si Alpaca registra movimientos con fecha anterior a la última vista.
    after = max(linked_on, last - timedelta(days=5)) if last else linked_on
    after -= timedelta(days=1)  # `after` de Alpaca es exclusivo

    try:
        activities = get_alpaca_cash_activities(*_alpaca_keys(account), after=after)
    except Exception as e:
        logger.error(f"No se pudieron leer los movimientos de capital de {account.display_name}: {e}")
        return 0

    known = set(flows.values_list("reference", flat=True))
    created = 0
    for act in activities:
        ref = f"alpaca:{act.get('id')}"
        if ref in known or str(act.get("status", "")).lower() == "canceled":
            continue
        amount = Decimal(str(act.get("net_amount") or "0"))
        if amount == 0:
            continue
        effective = date.fromisoformat(str(act.get("date"))[:10])
        if effective < linked_on:
            continue
        LedgerEntry.objects.create(
            portfolio=account.portfolio,
            broker_account=account,
            entry_type="deposit" if amount > 0 else "withdrawal",
            amount=amount,
            reference=ref,
            effective_date=effective,
        )
        known.add(ref)
        created += 1

    if created:
        logger.info(f"{created} movimientos de capital importados para {account.display_name}")
    return created


def sync_alpaca_account(account: BrokerAccount) -> dict:
    """Saldo + movimientos de capital. Devuelve el resultado del ping de saldo."""
    res = sync_alpaca_balance(account)
    if res.get("success"):
        res["new_cash_flows"] = sync_alpaca_cash_flows(account)
    return res
