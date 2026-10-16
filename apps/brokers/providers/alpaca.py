import requests
from datetime import date
from typing import Dict, Any, List, Optional

# Actividades no comerciales que mueven capital desde/hacia fuera de la cuenta:
# CSD depósito, CSW retiro, JNLC journal de efectivo entre cuentas, TRANS transferencia.
CASH_FLOW_ACTIVITY_TYPES = ("CSD", "CSW", "JNLC", "TRANS")


def _base_url(environment: str) -> str:
    return "https://paper-api.alpaca.markets" if environment == "paper" else "https://api.alpaca.markets"


def get_alpaca_cash_activities(
    api_key: str, secret_key: str, environment: str = "paper", after: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """
    Descarga de /v2/account/activities los depósitos, retiros y journals de efectivo
    (en orden cronológico, paginando con page_token). Lanza excepción si Alpaca responde con error.
    """
    headers = {"APCA-API-KEY-ID": api_key.strip(), "APCA-API-SECRET-KEY": secret_key.strip()}
    params: Dict[str, Any] = {
        "activity_types": ",".join(CASH_FLOW_ACTIVITY_TYPES),
        "direction": "asc",
        "page_size": 100,
    }
    if after:
        params["after"] = after.isoformat()

    activities: List[Dict[str, Any]] = []
    while True:
        res = requests.get(f"{_base_url(environment)}/v2/account/activities", headers=headers, params=params, timeout=15)
        res.raise_for_status()
        page = res.json() or []
        activities.extend(page)
        if len(page) < params["page_size"]:
            return activities
        params["page_token"] = page[-1]["id"]


def test_alpaca_credentials(api_key: str, secret_key: str, environment: str = "paper") -> Dict[str, Any]:
    """
    Tests Alpaca API credentials by pinging the /v2/account endpoint.
    Supports paper and live environments.
    """
    if not api_key or not secret_key:
        return {"success": False, "error": "API Key ID y Secret Key son requeridos."}

    base_url = "https://paper-api.alpaca.markets" if environment == "paper" else "https://api.alpaca.markets"
    url = f"{base_url}/v2/account"

    headers = {
        "APCA-API-KEY-ID": api_key.strip(),
        "APCA-API-SECRET-KEY": secret_key.strip(),
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json() if response.content else {}

        if response.status_code == 200:
            return {
                "success": True,
                "account_id": data.get("id"),
                "account_number": data.get("account_number"),
                "status": data.get("status"),
                "currency": data.get("currency", "USD"),
                "buying_power": float(data.get("buying_power", 0.0)),
                "cash": float(data.get("cash", 0.0)),
                "portfolio_value": float(data.get("portfolio_value", 0.0)),
                "trading_blocked": data.get("trading_blocked", False),
                "pattern_day_trader": data.get("pattern_day_trader", False),
            }
        elif response.status_code == 401 or response.status_code == 403:
            msg = data.get("message") or "Credenciales de Alpaca inválidas o no autorizadas."
            return {
                "success": False,
                "error": f"Error de autenticación Alpaca ({response.status_code}): {msg}",
            }
        else:
            msg = data.get("message") or response.text
            return {
                "success": False,
                "error": f"Error del servidor Alpaca ({response.status_code}): {msg}",
            }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Tiempo de espera agotado al conectar con los servidores de Alpaca."}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Error de conexión con Alpaca: {str(e)}"}
