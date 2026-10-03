"""Segment 3: score results against the answer key.

The answer key is read here and nowhere else. The matcher and the LLM step never see it.
"""
from __future__ import annotations

import csv
import re
from decimal import Decimal
from pathlib import Path

from recon.matcher import MATCHED, NEEDS_CORRECTION

MATCH_CATEGORIES = ["EXACT", "DATE_LAG", "NO_REFERENCE", "BATCH_DEPOSIT"]
CORRECTION_CATEGORIES = ["AMOUNT_ERROR", "TDS_DEDUCTED"]   # pairs the matcher flags NEEDS_CORRECTION
BOOK_ONLY_CATEGORIES = ["DUPLICATE_BOOK", "DEPOSIT_IN_TRANSIT", "OUTSTANDING_PAYMENT"]
BANK_ONLY_CATEGORIES = ["BANK_CHARGE", "INTEREST_CREDIT", "UNRECORDED_DEBIT", "DIRECT_CREDIT"]
EXCEPTION_CATEGORIES = CORRECTION_CATEGORIES + BOOK_ONLY_CATEGORIES + BANK_ONLY_CATEGORIES


def load_answer_key(path) -> list:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        items = list(csv.DictReader(handle))
    for item in items:
        item["book_ids"] = [i for i in item["book_ids"].split(";") if i]
        item["bank_ids"] = [i for i in item["bank_ids"].split(";") if i]
    return items


def _pair(book_ids, bank_ids):
    return frozenset(book_ids), frozenset(bank_ids)


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 1) if whole else 100.0


def score_matching(result, answer_key: list) -> dict:
    """How well the matcher did.

    match precision        share of the pairs it proposed that are exactly right
    auto-match rate        share of the items that should match which it matched correctly
    exceptions surfaced    share of the items that need action which it left for review
                           (or, for keying errors, paired and flagged NEEDS_CORRECTION)
    """
    proposed = {_pair(m.book_ids, m.bank_ids): m.status for m in result.matches}
    open_book = {line.book_id for line in result.unmatched_book}
    open_bank = {line.bank_id for line in result.unmatched_bank}
    expected_pairs, by_category = set(), {}
    for item in answer_key:
        pair, category = _pair(item["book_ids"], item["bank_ids"]), item["category"]
        if category in MATCH_CATEGORIES:
            expected_pairs.add((pair, MATCHED))
            correct = proposed.get(pair) == MATCHED
        elif category in CORRECTION_CATEGORIES:
            expected_pairs.add((pair, NEEDS_CORRECTION))
            correct = proposed.get(pair) == NEEDS_CORRECTION
        else:
            correct = set(item["book_ids"]) <= open_book and set(item["bank_ids"]) <= open_bank
        stats = by_category.setdefault(category, {"items": 0, "correct": 0})
        stats["items"] += 1
        stats["correct"] += int(correct)

    def total(categories, field):
        return sum(by_category.get(c, {}).get(field, 0) for c in categories)

    right_pairs = sum(1 for pair, status in proposed.items() if (pair, status) in expected_pairs)
    return {
        "items": len(answer_key),
        "match_precision_pct": _pct(right_pairs, len(proposed)),
        "auto_match_rate_pct": _pct(total(MATCH_CATEGORIES, "correct"), total(MATCH_CATEGORIES, "items")),
        "exceptions_surfaced_pct": _pct(total(EXCEPTION_CATEGORIES, "correct"), total(EXCEPTION_CATEGORIES, "items")),
        "proposed_pairs": len(proposed),
        "correct_pairs": right_pairs,
        "matchable_items": total(MATCH_CATEGORIES, "items"),
        "exception_items": total(EXCEPTION_CATEGORIES, "items"),
        "by_category": by_category,
    }


def format_report(scores: dict) -> str:
    lines = [f"Match precision      {scores['match_precision_pct']:5.1f}%",
             f"Auto-match rate      {scores['auto_match_rate_pct']:5.1f}%  of {scores['matchable_items']} matchable items",
             f"Exceptions surfaced  {scores['exceptions_surfaced_pct']:5.1f}%  of {scores['exception_items']} items needing action",
             "", f"{'category':<22}{'items':>6}{'correct':>9}"]
    for category, stats in scores["by_category"].items():
        lines.append(f"{category:<22}{stats['items']:>6}{stats['correct']:>9}")
    return "\n".join(lines)


def score_labels(suggestions: list, answer_key: list) -> dict:
    """How well the LLM step (or the rules baseline) labeled the exceptions.

    category accuracy   the suggested category equals the answer key's
    entry accuracy      the category is right AND the journal entry and adjustment match exactly
    """
    by_pair = {_pair(item["book_ids"], item["bank_ids"]): item for item in answer_key}
    by_category, right, entries_right = {}, 0, 0
    for s in suggestions:
        item = by_pair.get(_pair(s.book_ids, s.bank_ids))
        expected = item["category"] if item else "NOT_IN_ANSWER_KEY"
        correct = item is not None and s.category == expected
        entry_correct = (correct and s.journal_entry == item["suggested_entry"]
                         and s.adjustment == Decimal(item["adjustment"]))
        stats = by_category.setdefault(expected, {"items": 0, "correct": 0, "entry_correct": 0})
        stats["items"] += 1
        stats["correct"] += int(correct)
        stats["entry_correct"] += int(entry_correct)
        right += int(correct)
        entries_right += int(entry_correct)
    return {
        "exceptions": len(suggestions),
        "category_accuracy_pct": _pct(right, len(suggestions)),
        "entry_accuracy_pct": _pct(entries_right, len(suggestions)),
        "needs_review": sum(1 for s in suggestions if s.category == "NEEDS_REVIEW"),
        "by_category": by_category,
    }


# ---------------------------------------------------------------------------------------
# A simulated reviewer, for evaluating payee memory
# ---------------------------------------------------------------------------------------
def account_in_entry(entry: str) -> str:
    """The account opposite Bank: 'Dr Insurance Expense 9.00 / Cr Bank 9.00' -> 'Insurance Expense'."""
    debit, credit = (side.strip() for side in entry.split(" / "))
    side = credit if re.fullmatch(r"Dr Bank [\d.]+", debit) else debit
    return re.sub(r"^(?:Dr|Cr) | [\d.]+$", "", side)


def reviewer_approvals(suggestions: list, answer_key: list) -> list:
    """What a careful reviewer ends up approving for each bank-only line: the right category and
    account, whether the suggestion had them or the reviewer corrected it. Returns
    (bank_id, category, counter_account) tuples; the account is empty where it is fixed."""
    by_bank = {tuple(item["bank_ids"]): item for item in answer_key if item["bank_ids"] and not item["book_ids"]}
    approvals = []
    for s in suggestions:
        item = by_bank.get(tuple(s.bank_ids)) if s.kind == "BANK_ONLY" else None
        if item and item["category"] in BANK_ONLY_CATEGORIES:
            named = item["category"] in ("UNRECORDED_DEBIT", "DIRECT_CREDIT")
            approvals.append((s.bank_ids[0], item["category"],
                              account_in_entry(item["suggested_entry"]) if named else ""))
    return approvals
