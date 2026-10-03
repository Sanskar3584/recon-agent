"""Payee memory: reuse what a reviewer approved for a recurring bank line.

Auto-debits, bank charges, interest and many customer credits come back every month with the same
narration apart from reference numbers and dates. Once a reviewer has approved (or corrected) the
label for one of them, next month's line shouldn't be a fresh guess by keyword rules or a model.

The memory is keyed on the direction of the money and the narration's words that contain no digits,
and it stores the approved category and account. MemoryLabeler answers known lines from it and hands
everything else to the labeler it wraps, so a recurring line never reaches the LLM again. The
guardrails still check every remembered label, and every suggestion still waits for approval.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from recon.agent import BANK_ONLY, Label
from recon.loader import BankLine

REMEMBERED = {"BANK_CHARGE", "INTEREST_CREDIT", "UNRECORDED_DEBIT", "DIRECT_CREDIT"}
# Words that say how the money moved rather than who moved it. A key made only of these is too vague.
GENERIC_WORDS = {"NEFT", "RTGS", "IMPS", "UPI", "NACH", "ACH", "ECS", "CR", "DR", "CLG", "CHQ", "CHQS", "DEP",
                 "P2A", "TRF", "TO", "BY", "FROM", "PAID", "SLIP"}


def payee_key(line: BankLine) -> Optional[str]:
    """The direction plus the narration's words without digits: references, dates and month tags change
    every time, the payee's name doesn't. None when nothing specific to a payee is left."""
    words = [w for w in re.sub(r"[^A-Z0-9]", " ", line.narration.upper()).split()
             if not any(ch.isdigit() for ch in w)]
    if not [w for w in words if w not in GENERIC_WORDS]:
        return None
    return f"{'IN' if line.amount > 0 else 'OUT'} {' '.join(words)}"


@dataclass
class Remembered:
    category: str
    counter_account: str     # empty for charges and interest, whose accounts are fixed
    approved_on: str         # ISO date of the latest approval
    approvals: int           # approvals of this same label in a row
    example: str             # a narration it was learned from


class PayeeMemory:
    """Approved labels by payee, kept in a JSON file (or only in memory when no path is given)."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.entries = {}
        if self.path and self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = {key: Remembered(**value) for key, value in raw.items()}

    def recall(self, line: BankLine) -> Optional[Remembered]:
        key = payee_key(line)
        return self.entries.get(key) if key else None

    def remember(self, line: BankLine, category: str, counter_account: str, approved_on: str) -> bool:
        """Record an approved label; the latest approval wins. False if the line can't be remembered."""
        key = payee_key(line)
        if key is None or category not in REMEMBERED:
            return False
        old = self.entries.get(key)
        repeat = old is not None and (old.category, old.counter_account) == (category, counter_account)
        self.entries[key] = Remembered(category, counter_account, approved_on,
                                       old.approvals + 1 if repeat else 1, line.narration)
        return True

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {key: asdict(value) for key, value in sorted(self.entries.items())}
            self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class MemoryLabeler:
    """Labels bank-only lines from payee memory when it can, and passes everything else to `inner`."""

    def __init__(self, inner, memory: PayeeMemory):
        self.inner, self.memory = inner, memory
        self.name = f"{inner.name}+memory"
        self.hits = 0

    def label_all(self, cases: list) -> list:
        labels, rest = [None] * len(cases), []
        for i, case in enumerate(cases):
            hit = self.memory.recall(case.bank[0]) if case.kind == BANK_ONLY else None
            if hit is None:
                rest.append(i)
                continue
            labels[i] = Label(category=hit.category, counter_account=hit.counter_account, confidence=0.95,
                              explanation=f"Same payee as a line approved on {hit.approved_on} "
                                          f"({hit.approvals} approval(s) in a row); labeled the same way.")
            self.hits += 1
        if rest:
            answers = self.inner.label_all([cases[i] for i in rest])
            if len(answers) != len(rest):
                raise RuntimeError(f"{self.inner.name} returned {len(answers)} labels for {len(rest)} lines")
            for i, answer in zip(rest, answers):
                labels[i] = answer
        return labels
