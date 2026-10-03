"""Tax deducted at source (TDS): telling a TDS shortfall from a keying error.

When an Indian business pays a contractor, a professional or a landlord, it usually deducts TDS
and pays the rest. So a customer's payment can arrive short of the invoice, and a payment the
company makes to a vendor can leave the bank short of the bill. If the cash book shows the gross
amount, the matcher sees a shared reference with a different amount. That looks exactly like a
keying error, but it isn't one:

- on a receipt, the customer owes nothing more; the shortfall is a tax credit (TDS Receivable)
- on a payment, the company owes the shortfall to the government (TDS Payable)

Booking either one back to the customer's or vendor's account would be wrong. This module decides,
with arithmetic only, whether a shortfall fits TDS, and whether the two amounts instead look like
two swapped digits. No model is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

PAISA, RUPEE = Decimal("0.01"), Decimal("1")
GST = Decimal("0.18")          # TDS is usually worked out on the value before GST when GST is shown separately

# The common TDS rates, with the kind of payment each is used for.
RATES = {
    Decimal("0.001"): "purchase of goods",
    Decimal("0.01"): "contractors who are individuals",
    Decimal("0.02"): "contractors, technical services or equipment rent",
    Decimal("0.10"): "professional fees or property rent",
}


@dataclass(frozen=True)
class TdsFit:
    rate: Decimal              # 0.02 means 2%
    before_gst: bool           # True if the TDS was worked out on the value before 18% GST
    expected: Decimal          # the TDS that rate and base give (to the paisa, or rounded to the rupee)
    shortfall: Decimal         # how much less actually moved through the bank than the book shows

    @property
    def rate_text(self) -> str:
        return f"{(self.rate * 100).normalize():f}%"

    @property
    def base_text(self) -> str:
        return "value before 18% GST" if self.before_gst else "full amount"

    def describe(self) -> str:
        return (f"the shortfall of {self.shortfall:.2f} fits TDS at {self.rate_text} on the {self.base_text} "
                f"({self.expected:.2f}; usual for {RATES[self.rate]})")


def tds_fit(booked: Decimal, moved: Decimal) -> Optional[TdsFit]:
    """Does the gap between what the book shows and what moved through the bank fit TDS?

    The shortfall has to equal the TDS for one of the common rates exactly, either to the paisa
    or rounded to the nearest rupee, worked out on the full amount or on the value before 18% GST.
    TDS only ever makes the bank amount smaller, so a bank amount at or above the book's never fits.
    Signs are ignored, so this works for receipts and payments alike.
    """
    booked, moved = abs(Decimal(booked)), abs(Decimal(moved))
    if booked <= 0 or moved <= 0 or moved >= booked:
        return None
    shortfall = booked - moved
    for rate in RATES:
        for before_gst in (False, True):
            raw = (booked / (1 + GST) if before_gst else booked) * rate
            exact = raw.quantize(PAISA, ROUND_HALF_UP)
            # to the paisa, or to the rupee (rounded straight from the exact figure, or from the paise)
            for expected in (exact, raw.quantize(RUPEE, ROUND_HALF_UP), exact.quantize(RUPEE, ROUND_HALF_UP)):
                if shortfall == expected:
                    return TdsFit(rate, before_gst, expected.quantize(PAISA), shortfall)
    return None


def is_transposition(a: Decimal, b: Decimal) -> bool:
    """True when the two amounts are the same digits with two neighbours swapped (45,120 vs 45,210),
    the classic keying error."""
    x, y = f"{abs(Decimal(a)):.2f}", f"{abs(Decimal(b)):.2f}"
    if len(x) != len(y) or x == y:
        return False
    diff = [i for i in range(len(x)) if x[i] != y[i]]
    return (len(diff) == 2 and diff[1] == diff[0] + 1
            and x[diff[0]] == y[diff[1]] and x[diff[1]] == y[diff[0]])
