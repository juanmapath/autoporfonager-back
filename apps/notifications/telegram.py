import logging
import requests
from typing import List, Dict, Any
from django.conf import settings

logger = logging.getLogger(__name__)


def send_to_telegram(message: str, bot_token: str = "", chat_id: str = "") -> bool:
    token = bot_token or getattr(settings, "TELEGRAM_BOT_TOKEN", "")
    target_chat = chat_id or getattr(settings, "TELEGRAM_ALERT_CHAT_ID", "")

    if not token or not target_chat:
        logger.info(f"[Telegram Mock] (No credentials configured):\n{message}")
        return True

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": target_chat,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send telegram notification: {e}")
        return False


def format_manual_orders_message(
    portfolio_name: str,
    broker_name: str,
    rebalance_reason: str,
    orders: List[Dict[str, Any]],
) -> str:
    """Formats manual order notification matching MVP specification §6.4."""
    lines = [
        "🟡 *[ORDEN MANUAL] Portafolio:* " + portfolio_name,
        f"*Broker:* {broker_name}",
        f"*Rebalanceo:* {rebalance_reason}",
        "--------------------------------------------------",
    ]
    for o in orders:
        side = o.get("side", "BUY").upper()
        qty = o.get("qty", 0)
        symbol = o.get("symbol", "")
        price = o.get("price", 0.0)
        lines.append(f"- *{side}* {qty} {symbol}  (aprox ${price:,.2f})")
    lines.append("--------------------------------------------------")
    lines.append("Por favor ejecuta estas órdenes al cierre en tu broker.")
    return "\n".join(lines)
