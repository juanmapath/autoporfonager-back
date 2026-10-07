from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Union

QTY_PLACES = Decimal("0.00000001")    # 8 decimal places for shares/fractions
PRICE_PLACES = Decimal("0.000001")    # 6 decimal places for market prices
AMOUNT_PLACES = Decimal("0.01")       # 2 decimal places for cash/fiat balances
PERCENT_PLACES = Decimal("0.0001")    # 4 decimal places for weights (e.g. 0.2500)


def to_decimal(val: Union[float, int, str, Decimal, None], default: str = "0.00") -> Decimal:
    if val is None:
        return Decimal(default)
    if isinstance(val, Decimal):
        return val
    return Decimal(str(val))


def quantize_qty(val: Union[float, int, str, Decimal], round_down: bool = True) -> Decimal:
    d = to_decimal(val)
    rounding = ROUND_DOWN if round_down else ROUND_HALF_UP
    return d.quantize(QTY_PLACES, rounding=rounding)


def quantize_price(val: Union[float, int, str, Decimal]) -> Decimal:
    return to_decimal(val).quantize(PRICE_PLACES, rounding=ROUND_HALF_UP)


def quantize_amount(val: Union[float, int, str, Decimal]) -> Decimal:
    return to_decimal(val).quantize(AMOUNT_PLACES, rounding=ROUND_DOWN)


def quantize_weight(val: Union[float, int, str, Decimal]) -> Decimal:
    return to_decimal(val).quantize(PERCENT_PLACES, rounding=ROUND_HALF_UP)
