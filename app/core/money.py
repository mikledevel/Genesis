"""Публичные цены выражены в долларах, расчёты — в целых центах."""
from decimal import Decimal, InvalidOperation


def price_cents(value):
    try:
        amount = Decimal(str(value)) * 100
        if not amount.is_finite() or amount < 0 or amount > 100_000_000 or amount != amount.to_integral_value():
            raise ValueError
        return int(amount)
    except (ValueError, InvalidOperation, TypeError) as error:
        raise ValueError("Price must be nonnegative, at most $1,000,000, with at most two decimal places") from error
