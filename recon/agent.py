"""Segment 4: the LLM step. Labels every exception the matcher left and drafts the fix.

For each exception it builds an evidence pack (the line or lines, plus related facts such as
other book lines with the same reference), asks a labeler for a category, a counter account
and a short explanation, then applies guardrails:

- the category must fit the exception: a bank-only line can't be a deposit in transit, a
  payment can't be interest, and a shortfall can only be TDS if the arithmetic fits a TDS rate.
  Anything else becomes NEEDS_REVIEW
- counter accounts must come from the chart of accounts or the customer master
- the labeler never produces amounts. Python computes every adjustment and journal entry
- every suggestion starts as PENDING_APPROVAL; nothing is posted without a person

Labelers:
  RulesLabeler   keyword rules, no network: a baseline and an offline fallback
  LLMLabeler     LangChain structured output on OpenAI or a local Ollama model
  HybridLabeler  rules for amount pairs and book-only lines, the LLM for bank-only lines
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from recon.generator import journal_entry
from recon.loader import BankLine, BookLine
from recon.matcher import NEEDS_CORRECTION, MatchResult, normalize
from recon.tds import is_transposition, tds_fit

PAIR, BOOK_ONLY, BANK_ONLY = "PAIR", "BOOK_ONLY", "BANK_ONLY"
NEEDS_REVIEW = "NEEDS_REVIEW"
PENDING = "PENDING_APPROVAL"

EXPENSE_ACCOUNTS = ["Insurance Expense", "Lease Rental Expense", "Electricity Expense", "Rent Expense",
                    "Telephone Expense", "Professional Fees", "Loan Repayment", "Miscellaneous Expense"]
DEFAULT_MODELS = {"openai": "gpt-6-luna", "ollama": "llama3.1:8b"}


class Label(BaseModel):
    """What the labeler returns for one exception. Deliberately has no amount field."""
    category: Literal["AMOUNT_ERROR", "TDS_DEDUCTED", "DUPLICATE_BOOK", "DEPOSIT_IN_TRANSIT", "OUTSTANDING_PAYMENT",
                      "BANK_CHARGE", "INTEREST_CREDIT", "UNRECORDED_DEBIT", "DIRECT_CREDIT", "NEEDS_REVIEW"]
    counter_account: str = Field(default="", description=(
        "Only for UNRECORDED_DEBIT (an account from chart_of_accounts) and DIRECT_CREDIT "
        "('Customer Advances - ' plus a name from known_customers). Otherwise empty."))
    explanation: str = Field(description="One or two plain sentences for the reviewer.")
    confidence: float = Field(ge=0, le=1, description="How sure you are, from 0 to 1.")


@dataclass
class ExceptionCase:
    exception_id: str
    kind: str                 # PAIR, BOOK_ONLY or BANK_ONLY
    book: list
    bank: list
    evidence: dict

    @property
    def money_in(self) -> bool:
        lines = self.bank or self.book
        return lines[0].amount > 0


@dataclass
class Suggestion:
    exception_id: str
    kind: str
    category: str
    book_ids: list
    bank_ids: list
    line_date: date
    description: str
    amount: Decimal           # the line's signed amount (the bank's, for a pair)
    adjusts: str              # BOOK, BANK or NONE
    adjustment: Decimal
    counter_account: str
    journal_entry: str
    explanation: str
    confidence: float
    labeler: str
    guardrail: str = ""       # why the labeler's answer was overridden, if it was
    status: str = PENDING
    reviewer_note: str = ""


# ---------------------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------------------
def _book_view(line: BookLine) -> dict:
    return {"book_id": line.book_id, "date": line.date.isoformat(), "voucher_no": line.voucher_no,
            "party": line.party, "particulars": line.particulars, "mode": line.mode,
            "reference": line.reference, "direction": "money in" if line.amount > 0 else "money out",
            "amount": f"{abs(line.amount):.2f}"}


def _bank_view(line: BankLine) -> dict:
    return {"bank_id": line.bank_id, "value_date": line.value_date.isoformat(), "narration": line.narration,
            "bank_ref": line.bank_ref, "direction": "money in" if line.amount > 0 else "money out",
            "amount": f"{abs(line.amount):.2f}"}


def build_exceptions(result: MatchResult, book_lines: list, bank_lines: list, customers: list) -> list:
    """Turn everything the matcher couldn't clear into cases with evidence."""
    matched_to = {}
    for m in result.matches:
        for book_id in m.book_ids:
            matched_to[book_id] = m.bank_ids
    period_end = result.period_end
    cases = []

    def add(kind, book, bank, extra):
        evidence = {"exception_type": kind, "period_end": period_end.isoformat(),
                    "book_lines": [_book_view(l) for l in book], "bank_lines": [_bank_view(l) for l in bank]}
        evidence.update(extra)
        cases.append(ExceptionCase(f"EX-{len(cases) + 1:03d}", kind, book, bank, evidence))

    books_by_id = {l.book_id: l for l in book_lines}
    banks_by_id = {l.bank_id: l for l in bank_lines}
    for m in result.matches:
        if m.status == NEEDS_CORRECTION:
            fit = tds_fit(m.book_amount, m.bank_amount)
            add(PAIR, [books_by_id[i] for i in m.book_ids], [banks_by_id[i] for i in m.bank_ids],
                {"difference_bank_minus_book": f"{m.difference:.2f}", "matcher_note": m.detail,
                 "tds_check": fit.describe() if fit else "the difference does not fit any TDS rate",
                 "digits_swapped": is_transposition(m.book_amount, m.bank_amount)})
    for line in result.unmatched_book:
        twins = [l for l in book_lines if l.book_id != line.book_id and l.reference and l.reference == line.reference
                 and l.amount == line.amount]
        add(BOOK_ONLY, [line], [], {
            "days_before_period_end": (period_end - line.date).days,
            "same_reference_book_lines": [dict(_book_view(t), matched_to_bank_ids=matched_to.get(t.book_id, []))
                                          for t in twins]})
    for line in result.unmatched_bank:
        add(BANK_ONLY, [], [line], {"known_customers": customers, "chart_of_accounts": EXPENSE_ACCOUNTS})
    return cases


# ---------------------------------------------------------------------------------------
# Labelers
# ---------------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a bank reconciliation assistant. The matching engine compared a company's \
cash book with its bank statement and could not clear the exception below. It is one of:
- PAIR: a book line and a bank line share a reference but disagree on the amount
- BOOK_ONLY: a cash book line with no bank line
- BANK_ONLY: a bank line with no cash book line

Choose the category that best explains it:
- AMOUNT_ERROR (PAIR): the book amount was keyed in wrongly; the bank amount is right
- TDS_DEDUCTED (PAIR): less moved through the bank than the book shows because tax was deducted at \
source; only possible when tds_check says the shortfall fits a TDS rate
- DUPLICATE_BOOK (BOOK_ONLY): the same entry was booked twice; see same_reference_book_lines
- DEPOSIT_IN_TRANSIT (BOOK_ONLY, money in): booked near the period end, credited by the bank after it
- OUTSTANDING_PAYMENT (BOOK_ONLY, money out): paid or cheque issued, not yet debited by the bank
- BANK_CHARGE (BANK_ONLY, money out): a fee or charge debited by the bank
- INTEREST_CREDIT (BANK_ONLY, money in): interest credited by the bank
- UNRECORDED_DEBIT (BANK_ONLY, money out): an auto-debit, such as NACH, that the company never booked
- DIRECT_CREDIT (BANK_ONLY, money in): a customer paid straight into the account
- NEEDS_REVIEW: the evidence does not clearly support any category

counter_account: for UNRECORDED_DEBIT pick one account from chart_of_accounts; for DIRECT_CREDIT \
write "Customer Advances - " followed by the matching name copied exactly from known_customers. \
Leave it empty otherwise. Never state amounts; they are calculated separately. Keep the \
explanation to one or two plain sentences a reviewer can act on."""


class RulesLabeler:
    """Keyword rules. No network, fully predictable: the baseline the LLM is compared with."""
    name = "rules"

    def label_all(self, cases: list) -> list:
        return [self.label(case) for case in cases]

    def label(self, case: ExceptionCase) -> Label:
        if case.kind == PAIR:
            return self.label_pair(case)
        if case.kind == BOOK_ONLY:
            if case.evidence["same_reference_book_lines"]:
                twin = case.evidence["same_reference_book_lines"][0]
                return Label(category="DUPLICATE_BOOK", confidence=0.9,
                             explanation=f"Same reference and amount as {twin['voucher_no']}, which is already "
                                         f"matched; this looks like a second entry for one transaction.")
            days = case.evidence["days_before_period_end"]
            category = "DEPOSIT_IN_TRANSIT" if case.money_in else "OUTSTANDING_PAYMENT"
            return Label(category=category, confidence=0.8 if days <= 10 else 0.5,
                         explanation=f"Booked {days} day(s) before the period end with no bank posting yet; "
                                     f"expected to clear next period.")
        line = case.bank[0]
        text = line.narration.upper()
        if not case.money_in and re.search(r"CHRG|CHARGE|\bFEE\b", text):
            return Label(category="BANK_CHARGE", confidence=0.9, explanation="Bank charge not yet booked.")
        if case.money_in and re.search(r"\bINT\b|INTEREST", text):
            return Label(category="INTEREST_CREDIT", confidence=0.9, explanation="Interest credited by the bank.")
        if not case.money_in:
            for pattern, account in [(r"INSUR|LIFE", "Insurance Expense"), (r"LEAS", "Lease Rental Expense"),
                                     (r"POWER|ELECTRIC", "Electricity Expense"), (r"RENT|ESTATE", "Rent Expense")]:
                if re.search(pattern, text):
                    return Label(category="UNRECORDED_DEBIT", counter_account=account, confidence=0.8,
                                 explanation=f"Auto-debit not in the books; looks like {account.lower()}.")
            return Label(category="NEEDS_REVIEW", confidence=0.2, explanation="Unrecognized bank debit.")
        best, score = _closest_customer(text, case.evidence["known_customers"])
        if best and score >= 85:
            return Label(category="DIRECT_CREDIT", counter_account=f"Customer Advances - {best}", confidence=0.8,
                         explanation=f"Credit from {best} with no matching receipt in the books.")
        return Label(category="NEEDS_REVIEW", confidence=0.2, explanation="Unidentified bank credit.")

    @staticmethod
    def label_pair(case: ExceptionCase) -> Label:
        """A shared reference with a different amount: a keying error, or TDS deducted from the payment."""
        book, bank = case.book[0].amount, case.bank[0].amount
        fit, swapped = tds_fit(book, bank), is_transposition(book, bank)
        if fit and swapped:
            return Label(category="NEEDS_REVIEW", confidence=0.3,
                         explanation=f"Two explanations fit: {fit.describe()}, but the two amounts are also the "
                                     f"same digits with two swapped, which is how keying errors look. "
                                     f"Check the remittance advice.")
        if fit and case.money_in:
            return Label(category="TDS_DEDUCTED", confidence=0.85,
                         explanation=f"The customer paid less than the book shows: {fit.describe()}. They owe "
                                     f"nothing more; check it against their TDS certificate, then book it as "
                                     f"TDS Receivable.")
        if fit:
            return Label(category="TDS_DEDUCTED", confidence=0.85,
                         explanation=f"Less left the bank than the book shows: {fit.describe()}, deducted when "
                                     f"paying. Book it as TDS Payable and deposit it with the government.")
        return Label(category="AMOUNT_ERROR", confidence=0.9,
                     explanation="Book and bank share a reference but not the amount; the book entry "
                                 "was probably keyed in wrongly.")


def _closest_customer(text: str, customers: list):
    scored = [(max(fuzz.token_set_ratio(normalize(c), normalize(text)), fuzz.partial_ratio(normalize(c), normalize(text))), c)
              for c in customers]
    if not scored:
        return None, 0.0
    score, best = max(scored)
    return best, score


class LLMLabeler:
    """LangChain structured output on OpenAI (needs OPENAI_API_KEY) or a local Ollama model."""

    def __init__(self, provider: str = "openai", model: str | None = None, max_concurrency: int = 4):
        try:
            from dotenv import load_dotenv
            load_dotenv()                       # picks up OPENAI_API_KEY / RECON_LLM_MODEL from .env
        except ImportError:
            pass
        model = model or os.getenv("RECON_LLM_MODEL") or DEFAULT_MODELS.get(provider)
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            chat = ChatOpenAI(model=model)
            self.chain = chat.with_structured_output(Label, method="function_calling")
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            chat = ChatOllama(model=model, temperature=0)
            self.chain = chat.with_structured_output(Label)
            max_concurrency = 1                 # one local model, one request at a time
        else:
            raise ValueError(f"Unknown provider {provider!r}; use openai or ollama")
        self.name = f"{provider}:{model}"
        self.max_concurrency = max_concurrency

    def label_all(self, cases: list) -> list:
        prompts = [[("system", SYSTEM_PROMPT), ("human", json.dumps(case.evidence, indent=1))] for case in cases]
        return self.chain.batch(prompts, config={"max_concurrency": self.max_concurrency}, return_exceptions=True)


class HybridLabeler:
    """Rules where the data already decides, the LLM only where a narration has to be read.

    Amount pairs and book-only lines are settled by structured facts: the direction of the money
    and whether an identical entry already cleared the bank. The rules get those right every time.
    Bank-only lines (charges, auto-debits, credits from customers) can only be told apart by reading
    the bank's free text, so only they go to the LLM.
    """

    def __init__(self, llm, rules=None):
        self.llm, self.rules = llm, rules or RulesLabeler()
        self.name = f"hybrid:{llm.name}"

    def label_all(self, cases: list) -> list:
        labels = [None if case.kind == BANK_ONLY else self.rules.label(case) for case in cases]
        to_llm = [i for i, case in enumerate(cases) if case.kind == BANK_ONLY]
        if to_llm:
            answers = self.llm.label_all([cases[i] for i in to_llm])
            if len(answers) != len(to_llm):
                raise RuntimeError(f"LLM returned {len(answers)} labels for {len(to_llm)} lines")
            for i, answer in zip(to_llm, answers):
                labels[i] = answer
        return labels


# ---------------------------------------------------------------------------------------
# Guardrails and the amounts Python computes
# ---------------------------------------------------------------------------------------
def allowed_categories(case: ExceptionCase) -> set:
    if case.kind == PAIR:
        if tds_fit(case.book[0].amount, case.bank[0].amount):
            return {"AMOUNT_ERROR", "TDS_DEDUCTED"}
        return {"AMOUNT_ERROR"}
    if case.kind == BOOK_ONLY:
        return {"DUPLICATE_BOOK", "DEPOSIT_IN_TRANSIT" if case.money_in else "OUTSTANDING_PAYMENT"}
    return {"INTEREST_CREDIT", "DIRECT_CREDIT"} if case.money_in else {"BANK_CHARGE", "UNRECORDED_DEBIT"}


def _ledger(line: BookLine) -> str:
    return f"{'Accounts Receivable' if line.amount > 0 else 'Accounts Payable'} - {line.party}"


def resolve(case: ExceptionCase, label, labeler_name: str) -> Suggestion:
    """Apply the guardrails to one label and compute the adjustment and journal entry."""
    guardrail = ""
    if isinstance(label, Exception):
        guardrail = f"labeler failed: {type(label).__name__}: {str(label)[:200]}"
        label = Label(category=NEEDS_REVIEW, explanation="The labeler could not process this line.", confidence=0)
    category = label.category
    if category != NEEDS_REVIEW and category not in allowed_categories(case):
        if category == "TDS_DEDUCTED" and case.kind == PAIR:
            guardrail = "TDS_DEDUCTED needs a shortfall that fits a TDS rate, and this one doesn't"
        else:
            guardrail = f"{category} is not possible for a {case.kind} line going {'in' if case.money_in else 'out'}"
        category = NEEDS_REVIEW

    counter = ""
    if category == "UNRECORDED_DEBIT":
        counter = label.counter_account.strip()
        if counter not in EXPENSE_ACCOUNTS:
            guardrail, category = f"account {counter!r} is not in the chart of accounts", NEEDS_REVIEW
    elif category == "DIRECT_CREDIT":
        counter = label.counter_account.strip()
        valid = {f"Customer Advances - {c}" for c in case.evidence.get("known_customers", [])}
        if counter not in valid:
            guardrail, category = f"{counter!r} does not name a customer from the master list", NEEDS_REVIEW

    book_total = sum((l.amount for l in case.book), Decimal("0.00"))
    bank_total = sum((l.amount for l in case.bank), Decimal("0.00"))
    adjusts, adjustment, entry = "NONE", Decimal("0.00"), "Needs review: no entry suggested"
    if category == "AMOUNT_ERROR":
        adjusts, adjustment, counter = "BOOK", bank_total - book_total, _ledger(case.book[0])
    elif category == "TDS_DEDUCTED":
        adjusts, adjustment = "BOOK", bank_total - book_total
        counter = "TDS Receivable" if case.money_in else "TDS Payable"
    elif category == "DUPLICATE_BOOK":
        adjusts, adjustment, counter = "BOOK", -book_total, _ledger(case.book[0])
    elif category in ("DEPOSIT_IN_TRANSIT", "OUTSTANDING_PAYMENT"):
        adjusts, adjustment, counter = "BANK", book_total, ""
        entry = "No entry needed: timing difference"
    elif category == "BANK_CHARGE":
        adjusts, adjustment, counter = "BOOK", bank_total, "Bank Charges"
    elif category == "INTEREST_CREDIT":
        adjusts, adjustment, counter = "BOOK", bank_total, "Interest Income"
    elif category in ("UNRECORDED_DEBIT", "DIRECT_CREDIT"):
        adjusts, adjustment = "BOOK", bank_total
    if adjusts == "BOOK":
        entry = journal_entry(adjustment, counter)

    line = (case.bank or case.book)[0]
    return Suggestion(
        exception_id=case.exception_id, kind=case.kind, category=category,
        book_ids=[l.book_id for l in case.book], bank_ids=[l.bank_id for l in case.bank],
        line_date=line.value_date if isinstance(line, BankLine) else line.date,
        description=line.narration if isinstance(line, BankLine) else line.particulars,
        amount=bank_total if case.bank else book_total, adjusts=adjusts, adjustment=adjustment,
        counter_account=counter, journal_entry=entry, explanation=label.explanation.strip(),
        confidence=float(label.confidence), labeler=labeler_name, guardrail=guardrail)


def label_exceptions(cases: list, labeler) -> list:
    labels = labeler.label_all(cases)
    return [resolve(case, label, labeler.name) for case, label in zip(cases, labels)]
