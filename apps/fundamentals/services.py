import logging
from datetime import timedelta
from typing import Dict, Iterable, Optional

from django.conf import settings
from django.db.models import Sum
from django.utils import timezone

from apps.instruments.models import Instrument
from apps.fundamentals.models import FundamentalStatement, FundamentalFetchLog
from apps.fundamentals.providers import ProviderChain, estimate_available_at

logger = logging.getLogger(__name__)

# Only equities have meaningful fundamentals; ETFs/crypto/index are neutral in rankings.
FUNDAMENTAL_ASSET_CLASSES = ("equity",)
REFRESH_AFTER_DAYS = 7


def fmp_calls_today() -> int:
    start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    agg = FundamentalFetchLog.objects.filter(source="fmp", created_at__gte=start).aggregate(s=Sum("calls_used"))
    return agg["s"] or 0


def needs_refresh(inst: Instrument) -> bool:
    last = FundamentalStatement.objects.filter(instrument=inst).order_by("-updated_at").first()
    if not last:
        return True
    return last.updated_at < timezone.now() - timedelta(days=REFRESH_AFTER_DAYS)


def sync_instrument(inst: Instrument, chain: Optional[ProviderChain] = None) -> Dict:
    chain = chain or ProviderChain()
    if fmp_calls_today() + 3 > getattr(settings, "FMP_DAILY_CALL_BUDGET", 240):
        chain.skip.add("fmp")

    symbol = inst.yahoo_symbol or inst.symbol
    res, attempts = chain.fetch(symbol)
    for a in attempts:
        FundamentalFetchLog.objects.create(
            instrument=inst, source=a["source"], status=a["status"],
            calls_used=a["calls"], error=a["error"],
            statements_saved=len(res.statements) if (res and a["status"] == "ok") else 0,
        )
    if not res:
        return {"symbol": inst.symbol, "saved": 0, "attempts": attempts}

    saved = 0
    for st in res.statements:
        FundamentalStatement.objects.update_or_create(
            instrument=inst,
            period_type=st.period_type,
            period_end=st.period_end,
            defaults={
                "available_at": st.available_at or estimate_available_at(st.period_end, st.period_type),
                "source": res.source,
                "data": st.data,
                "raw": _json_safe(st.raw),
            },
        )
        saved += 1
    return {"symbol": inst.symbol, "saved": saved, "source": res.source, "attempts": attempts}


def instruments_for_active_ranked_bots() -> Iterable[Instrument]:
    from apps.strategies.models import StrategyInstrument
    ids = StrategyInstrument.objects.filter(
        version__status="live",
        version__strategy__is_active=True,
        version__strategy__kind="ranked_allocation",
        role="traded",
        instrument__asset_class__in=FUNDAMENTAL_ASSET_CLASSES,
    ).values_list("instrument_id", flat=True).distinct()
    return Instrument.objects.filter(id__in=list(ids))


def sync_active_universe(force: bool = False) -> Dict:
    chain = ProviderChain()
    results = []
    for inst in instruments_for_active_ranked_bots():
        if not force and not needs_refresh(inst):
            continue
        try:
            results.append(sync_instrument(inst, chain))
        except Exception as e:  # noqa: BLE001
            logger.error("Fundamental sync failed for %s: %s", inst.symbol, e)
            results.append({"symbol": inst.symbol, "saved": 0, "error": str(e)})
    return {"synced": len(results), "results": results}


def _json_safe(v):
    import json
    try:
        return json.loads(json.dumps(v, default=str))
    except Exception:  # noqa: BLE001
        return {}
