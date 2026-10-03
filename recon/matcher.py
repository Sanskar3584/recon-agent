"""Segment 2: the matching engine.

Pairs cash book lines with bank statement lines in four passes, strongest evidence first.
Each line is used at most once; whatever is left becomes an exception for the LLM step.

  1. reference    a shared reference (UTR, cheque number, invoice number) and the same amount
  2. amount_diff  a shared reference but a different amount. Probably a keying error, so the
                  pair is flagged NEEDS_CORRECTION instead of being accepted
  3. amount_date  the same amount within a few days; the closest counterparty name wins ties
  4. one_to_many  several book lines that add up to one bank line (cheques on one deposit slip)
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from itertools import combinations

from rapidfuzz import fuzz

from recon.loader import BankLine, BookLine

MATCHED = "MATCHED"
NEEDS_CORRECTION = "NEEDS_CORRECTION"


@dataclass(frozen=True)
class MatchSettings:
    max_days_late: int = 16        # calendar days the bank may post after the book (late cheques)
    max_days_early: int = 1        # calendar days the bank may post before the book
    amount_date_days: int = 5      # window for the amount + date and one-to-many passes
    min_reference_length: int = 4  # shorter references are too weak to count as evidence
    max_group_size: int = 6        # most book lines one bank line may combine
    max_group_candidates: int = 20 # cap on lines searched per bank line, to bound the work


@dataclass
class Match:
    match_id: str
    rule: str                  # which pass found it
    status: str                # MATCHED or NEEDS_CORRECTION
    book_ids: list
    bank_ids: list
    book_amount: Decimal
    bank_amount: Decimal
    score: float               # 0-100, how strong the evidence is
    detail: str

    @property
    def difference(self) -> Decimal:
        return self.bank_amount - self.book_amount


@dataclass
class MatchResult:
    matches: list
    unmatched_book: list
    unmatched_bank: list
    period_end: date

    def summary(self) -> dict:
        matched = [m for m in self.matches if m.status == MATCHED]
        matched_lines = sum(len(m.book_ids) + len(m.bank_ids) for m in matched)
        total_lines = (sum(len(m.book_ids) + len(m.bank_ids) for m in self.matches)
                       + len(self.unmatched_book) + len(self.unmatched_bank))
        by_rule = {}
        for m in self.matches:
            by_rule[m.rule] = by_rule.get(m.rule, 0) + 1
        return {
            "period_end": self.period_end.isoformat(),
            "matches": len(matched),
            "needs_correction": len(self.matches) - len(matched),
            "unmatched_book_lines": len(self.unmatched_book),
            "unmatched_bank_lines": len(self.unmatched_bank),
            "lines_auto_matched_pct": round(100 * matched_lines / total_lines, 1) if total_lines else 0.0,
            "matches_by_rule": by_rule,
        }


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^A-Z0-9]", " ", text.upper()).split())


def shares_reference(book: BookLine, bank: BankLine, min_length: int = 4) -> bool:
    reference = book.reference.strip().upper()
    if len(reference) < min_length:
        return False
    if reference == bank.bank_ref.strip().upper():
        return True
    pattern = rf"(?<![A-Z0-9]){re.escape(reference)}(?![A-Z0-9])"   # whole token, not part of one
    return re.search(pattern, bank.narration.upper()) is not None


def name_score(book: BookLine, bank: BankLine) -> float:
    party, narration = normalize(book.party), normalize(bank.narration)
    if not party:
        return 0.0
    return float(max(fuzz.token_set_ratio(party, narration), fuzz.partial_ratio(party, narration)))


def days_apart(book: BookLine, bank: BankLine) -> int:
    return (bank.value_date - book.date).days


def same_direction(book: BookLine, bank: BankLine) -> bool:
    return (book.amount > 0) == (bank.amount > 0)


def period_end_of(book_lines: list, bank_lines: list) -> date:
    latest = max([l.value_date for l in bank_lines] or [l.date for l in book_lines])
    return date(latest.year, latest.month, calendar.monthrange(latest.year, latest.month)[1])


class _Matcher:
    def __init__(self, book_lines: list, bank_lines: list, settings: MatchSettings):
        self.s = settings
        self.book = {l.book_id: l for l in sorted(book_lines, key=lambda l: (l.date, l.book_id))}
        self.bank = {l.bank_id: l for l in sorted(bank_lines, key=lambda l: (l.value_date, l.bank_id))}
        self.matches = []

    def record(self, rule, status, books, banks, score, detail):
        for line in books:
            del self.book[line.book_id]
        for line in banks:
            del self.bank[line.bank_id]
        self.matches.append(Match(
            match_id=f"M-{len(self.matches) + 1:04d}", rule=rule, status=status,
            book_ids=[l.book_id for l in books], bank_ids=[l.bank_id for l in banks],
            book_amount=sum((l.amount for l in books), Decimal("0.00")),
            bank_amount=sum((l.amount for l in banks), Decimal("0.00")),
            score=round(score, 1), detail=detail))

    def in_window(self, book, bank) -> bool:
        return -self.s.max_days_early <= days_apart(book, bank) <= self.s.max_days_late

    def pass_reference(self) -> None:
        for book in list(self.book.values()):
            candidates = [bank for bank in self.bank.values()
                          if same_direction(book, bank) and bank.amount == book.amount
                          and self.in_window(book, bank)
                          and shares_reference(book, bank, self.s.min_reference_length)]
            if candidates:
                bank = min(candidates, key=lambda k: (abs(days_apart(book, k)), k.bank_id))
                self.record("reference", MATCHED, [book], [bank], 100.0,
                            f"shared reference {book.reference}")

    def pass_amount_diff(self) -> None:
        for book in list(self.book.values()):
            candidates = [bank for bank in self.bank.values()
                          if same_direction(book, bank) and self.in_window(book, bank)
                          and shares_reference(book, bank, self.s.min_reference_length)]
            if len(candidates) == 1:
                bank = candidates[0]
                self.record("amount_diff", NEEDS_CORRECTION, [book], [bank], 90.0,
                            f"shared reference {book.reference}, but the book shows "
                            f"{abs(book.amount):.2f} and the bank {abs(bank.amount):.2f}")

    def pass_amount_date(self) -> None:
        pairs = []
        for book in self.book.values():
            for bank in self.bank.values():
                lag = days_apart(book, bank)
                if (same_direction(book, bank) and bank.amount == book.amount
                        and 0 <= lag <= self.s.amount_date_days):
                    pairs.append((name_score(book, bank), lag, book, bank))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2].book_id, p[3].bank_id))
        for score, lag, book, bank in pairs:
            if book.book_id in self.book and bank.bank_id in self.bank:
                self.record("amount_date", MATCHED, [book], [bank], score,
                            f"same amount, {lag} day(s) apart, name similarity {score:.0f}")

    def pass_one_to_many(self) -> None:
        for bank in list(self.bank.values()):
            candidates = [book for book in self.book.values()
                          if same_direction(book, bank) and 0 <= days_apart(book, bank) <= self.s.amount_date_days
                          and abs(book.amount) < abs(bank.amount)][: self.s.max_group_candidates]
            by_day = {}
            for book in candidates:
                by_day.setdefault(book.date, []).append(book)
            pools = list(by_day.values()) + [candidates]   # one deposit slip first, then the whole window
            for pool in pools:
                group = self.find_group(pool, bank.amount)
                if group:
                    self.record("one_to_many", MATCHED, group, [bank], 95.0,
                                f"{len(group)} book lines add up to the bank amount")
                    break

    def find_group(self, pool: list, target: Decimal):
        for size in range(2, min(self.s.max_group_size, len(pool)) + 1):
            for combo in combinations(pool, size):
                if sum((l.amount for l in combo), Decimal("0.00")) == target:
                    return list(combo)
        return None

    def run(self, book_lines, bank_lines) -> MatchResult:
        self.pass_reference()
        self.pass_amount_diff()
        self.pass_amount_date()
        self.pass_one_to_many()
        return MatchResult(self.matches, list(self.book.values()), list(self.bank.values()),
                           period_end_of(book_lines, bank_lines))


def match(book_lines: list, bank_lines: list, settings: MatchSettings = MatchSettings()) -> MatchResult:
    """Run the four passes and return the matches plus everything left unmatched."""
    return _Matcher(book_lines, bank_lines, settings).run(book_lines, bank_lines)
