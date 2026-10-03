"""Segment 1: read the cash book and the bank statement.

Every field is read as text first, so references such as cheque number 057202 keep their
leading zeros. Amounts become signed Decimals from the company's point of view on both
sides: money in is positive, money out is negative.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

ZERO = Decimal("0.00")
BOOK_COLUMNS = ["book_id", "date", "voucher_no", "party", "particulars", "mode", "reference",
                "receipt", "payment"]
BANK_COLUMNS = ["bank_id", "value_date", "narration", "bank_ref", "deposit", "withdrawal"]


class LoadError(ValueError):
    """A file is missing a column or holds a value that can't be read."""


@dataclass(frozen=True)
class BookLine:
    book_id: str
    date: date
    voucher_no: str
    party: str
    particulars: str
    mode: str
    reference: str
    amount: Decimal  # + receipt, - payment


@dataclass(frozen=True)
class BankLine:
    bank_id: str
    value_date: date
    narration: str
    bank_ref: str
    amount: Decimal  # + deposit, - withdrawal


def _rows(path, required):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:   # -sig: Excel adds a BOM
        reader = csv.DictReader(handle)
        missing = [column for column in required if column not in (reader.fieldnames or [])]
        if missing:
            raise LoadError(f"{path.name}: missing columns {missing}")
        for line_no, row in enumerate(reader, start=2):             # line 1 is the header
            yield f"{path.name} line {line_no}", row


def _money(text: str, where: str) -> Decimal:
    text = (text or "").replace(",", "").strip()                    # Excel may add thousands separators
    try:
        return Decimal(text) if text else ZERO
    except InvalidOperation:
        raise LoadError(f"{where}: {text!r} is not an amount") from None


def _signed(money_in: str, money_out: str, where: str) -> Decimal:
    amount_in, amount_out = _money(money_in, where), _money(money_out, where)
    if amount_in < 0 or amount_out < 0:
        raise LoadError(f"{where}: amounts must not be negative")
    if (amount_in == 0) == (amount_out == 0):
        raise LoadError(f"{where}: exactly one of the two amount columns must be filled")
    return amount_in - amount_out


def _date(text: str, where: str) -> date:
    try:
        return date.fromisoformat((text or "").strip())
    except ValueError:
        raise LoadError(f"{where}: {text!r} is not a YYYY-MM-DD date") from None


def _unique(ids: list, path) -> None:
    seen, dupes = set(), set()
    for line_id in ids:
        (dupes if line_id in seen else seen).add(line_id)
    if dupes:
        raise LoadError(f"{Path(path).name}: repeated ids {sorted(dupes)}")


def load_book(path) -> list:
    lines = [BookLine(book_id=row["book_id"].strip(), date=_date(row["date"], where),
                      voucher_no=row["voucher_no"].strip(), party=row["party"].strip(),
                      particulars=row["particulars"].strip(), mode=row["mode"].strip().upper(),
                      reference=row["reference"].strip(),
                      amount=_signed(row["receipt"], row["payment"], where))
             for where, row in _rows(path, BOOK_COLUMNS)]
    _unique([line.book_id for line in lines], path)
    return lines


def load_bank(path) -> list:
    lines = [BankLine(bank_id=row["bank_id"].strip(), value_date=_date(row["value_date"], where),
                      narration=row["narration"].strip(), bank_ref=row["bank_ref"].strip(),
                      amount=_signed(row["deposit"], row["withdrawal"], where))
             for where, row in _rows(path, BANK_COLUMNS)]
    _unique([line.bank_id for line in lines], path)
    return lines


def load_parties(path) -> dict:
    """Customer and vendor master: {party name: CUSTOMER or VENDOR}."""
    return {row["party"].strip(): row["type"].strip().upper() for _, row in _rows(path, ["party", "type"])}


def opening_balance(path, money_in: str, money_out: str) -> Decimal:
    """Opening balance implied by the first row: its running balance minus its own amount."""
    for where, row in _rows(path, ["balance", money_in, money_out]):
        return _money(row["balance"], where) - _signed(row[money_in], row[money_out], where)
    return ZERO
