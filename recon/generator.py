"""Synthetic cash book and bank statement generator (step 1 of the reconciliation agent).

It produces one month of a company's cash book (the company's own record of its bank
account) and the bank's statement for the same account, with the differences a real bank
reconciliation has to explain: clearing delays, deposits in transit, uncleared cheques, bank
charges, interest, auto-debits nobody booked, batched cheque deposits, keying errors,
duplicate entries and bank narrations with no usable reference.

Every difference is injected on purpose and written to an answer key, so the matching
engine and the LLM agent built next can be scored against the right answers. The matcher
and the agent must never read the answer key; it exists only for scoring.

Money is held as Decimal, so every total is exact to the paisa.

Usage:
    python -m recon.generator --month 2026-06 --seed 42 --out data
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Optional

PAISA = Decimal("0.01")
ZERO = Decimal("0.00")

# ---------------------------------------------------------------------------------------
# Reconciliation categories
# ---------------------------------------------------------------------------------------
# On both sides and agreeing (no adjustment needed):
EXACT = "EXACT"                  # same amount, same day, shared reference
DATE_LAG = "DATE_LAG"            # same amount and reference; the bank posted it later
NO_REFERENCE = "NO_REFERENCE"    # same amount, but the bank narration has no shared reference
BATCH_DEPOSIT = "BATCH_DEPOSIT"  # several cheques in the book, one combined credit at the bank
# On both sides, but the book is wrong:
AMOUNT_ERROR = "AMOUNT_ERROR"    # two digits transposed when the entry was keyed in
TDS_DEDUCTED = "TDS_DEDUCTED"    # settled net of tax deducted at source; the book shows the gross (opt-in)
# Only in the book:
DUPLICATE_BOOK = "DUPLICATE_BOOK"            # the same entry keyed twice
DEPOSIT_IN_TRANSIT = "DEPOSIT_IN_TRANSIT"    # receipt booked this month, credited next month
OUTSTANDING_PAYMENT = "OUTSTANDING_PAYMENT"  # payment booked this month, not yet debited
# Only at the bank:
BANK_CHARGE = "BANK_CHARGE"
INTEREST_CREDIT = "INTEREST_CREDIT"
UNRECORDED_DEBIT = "UNRECORDED_DEBIT"  # an auto-debit (NACH) the company never booked
DIRECT_CREDIT = "DIRECT_CREDIT"        # a customer paid straight into the account

CATEGORIES = [EXACT, DATE_LAG, NO_REFERENCE, BATCH_DEPOSIT, AMOUNT_ERROR, DUPLICATE_BOOK,
              DEPOSIT_IN_TRANSIT, OUTSTANDING_PAYMENT, BANK_CHARGE, INTEREST_CREDIT,
              UNRECORDED_DEBIT, DIRECT_CREDIT]
OPTIONAL_CATEGORIES = [TDS_DEDUCTED]   # generated only when the config asks, so the default months never change
MATCHED = {EXACT, DATE_LAG, NO_REFERENCE, BATCH_DEPOSIT}
BOOK_SIDE = {AMOUNT_ERROR, TDS_DEDUCTED, DUPLICATE_BOOK, BANK_CHARGE, INTEREST_CREDIT, UNRECORDED_DEBIT,
             DIRECT_CREDIT}
BANK_SIDE = {DEPOSIT_IN_TRANSIT, OUTSTANDING_PAYMENT}

# ---------------------------------------------------------------------------------------
# Fictional parties
# ---------------------------------------------------------------------------------------
CUSTOMERS = [
    "Arjun Textiles Pvt Ltd", "Meera Foods Pvt Ltd", "Sahyadri Engineering Works",
    "Nilgiri Tea Traders", "Vishwa Auto Components", "Ganga Pharma Distributors",
    "Rangoli Home Decor", "Deccan Steel Fabricators", "Kaveri Agro Products",
    "Tulsi Healthcare LLP", "Palash Motors Service Centre", "Chandan Paints Depot",
    "Shivalik Hardware Stores", "Prakash Electricals", "Elaichi Spices Exports",
    "Aravalli Constructions", "Konkan Marine Supplies", "Narmada Plastics",
    "Vindhya Cement Traders", "Nakshatra IT Solutions",
]
VENDORS = [
    "Shakti Logistics", "Kaveri Packaging Co", "Omkar Steel Suppliers", "Pragati Printers",
    "Vayu Freight Movers", "Annapurna Canteen Services", "Surya Power Solutions",
    "Mahalaxmi Chemicals", "Swastik Tools and Dies", "Jyoti Office Supplies",
    "Anmol Castings", "Trimurti Security Services", "Saraswati Stationers",
    "Ekta Electrical Contractors", "Pushpak Couriers",
]
COMPANY = "Kestrel Components Pvt Ltd"   # the fictional company whose books we reconcile
BANK_CODES = ["NBXA", "QRTB", "ZENB", "FNXB", "PLMB", "VRDB"]  # fictional bank codes for UTRs
BANK_CHARGES = [  # (narration, amount including 18% GST)
    ("NEFT/RTGS TXN CHRGS", "236.00"), ("SMS ALERT CHRGS", "59.00"),
    ("CHQ BOOK ISSUE CHRGS", "295.00"), ("ACCOUNT MAINT CHRGS", "590.00"),
    ("STOP PAYMENT CHRGS", "118.00"), ("DEBIT CARD ANNUAL FEE", "708.00"),
]
TRICKY_BANK_CHARGES = [("LOCKER RENT FY26-27", "3540.00"), ("DD COMMISSION", "177.00")]  # no "charges" keyword
UNRECORDED_DEBITS = [  # (payee, account the company should book it to, amount)
    ("Suraksha Life Insurance", "Insurance Expense", "18540.00"),
    ("Vistar Equipment Leasing", "Lease Rental Expense", "42000.00"),
    ("Prakash Power Distribution", "Electricity Expense", "27315.00"),
]
TRICKY_UNRECORDED_DEBITS = [
    ("Aarogya Health Plan", "Insurance Expense", "12650.00"),
    ("Tarang Broadband Services", "Telephone Expense", "2360.00"),
    ("Nyay Legal Associates", "Professional Fees", "35400.00"),
]


@dataclass
class GeneratorConfig:
    month: str = "2026-06"                     # statement period, YYYY-MM
    seed: int = 42
    opening_balance: Decimal = Decimal("5000000.00")
    n_receipts: int = 130                      # ordinary customer receipts
    n_payments: int = 100                      # ordinary vendor payments
    n_batches: int = 4                         # deposit slips with 2-4 cheques each
    no_reference_rate: float = 0.12            # share of eligible items with no usable reference
    amount_error_rate: float = 0.03            # share of eligible items keyed in wrongly
    n_duplicates: int = 2
    n_bank_charges: int = 5
    n_unrecorded_debits: int = 2
    n_direct_credits: int = 2
    interest_credit: bool = True
    n_lookalike_groups: int = 3                # same-amount, same-day receipts with no reference
    tricky_narrations: bool = True             # bank-only lines that simple keyword rules misread
    n_tds: int = 0                             # receipts and payments settled net of TDS (opt-in scenario)


@dataclass
class Event:
    """One movement of money, or (for DUPLICATE_BOOK) one bookkeeping mistake."""
    kind: str                           # "RECEIPT" (money in) or "PAYMENT" (money out)
    mode: str                           # NEFT, RTGS, IMPS, UPI, CHEQUE, NACH, NETBANKING, CHARGE, INTEREST
    counterparty: str
    amount: Decimal                     # the true amount, always positive
    book_date: Optional[date] = None    # None if the company never booked it
    bank_date: Optional[date] = None    # None if it never reaches the bank
    doc: str = ""                       # invoice or bill number
    bank_ref: str = ""                  # UTR, RRN, cheque number, CPIN...
    book_amount: Optional[Decimal] = None  # what the company keyed in (differs for AMOUNT_ERROR)
    particulars: str = ""               # preset book text for recurring items
    reference: str = ""                 # preset book reference for recurring items
    narration: str = ""                 # preset bank text for recurring and bank-only items
    batch: str = ""                     # deposit slip number for batched cheques
    no_reference: bool = False
    recurring: bool = False
    category: str = ""
    other_account: str = ""             # the account on the other side of the adjusting entry
    duplicate_of: Optional["Event"] = None
    order: int = 0                      # creation order, a stable tie-breaker when sorting
    tds_rate: Optional[Decimal] = None  # for TDS_DEDUCTED: the rate deducted
    tds_before_gst: bool = False        # for TDS_DEDUCTED: worked out on the value before 18% GST
    book_id: str = ""
    bank_id: str = ""
    voucher: str = ""

    @property
    def sign(self) -> int:
        return 1 if self.kind == "RECEIPT" else -1


@dataclass
class GeneratedData:
    config: GeneratorConfig
    book: list          # cash book lines (dicts)
    bank: list          # bank statement lines (dicts)
    answer_key: list    # reconciliation items (dicts)
    summary: dict       # the bank reconciliation statement


# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------
def business_days(start: date, end: date) -> list:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:          # Monday-Friday; bank holidays are ignored
            days.append(d)
        d += timedelta(days=1)
    return days


def add_business_days(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def bank_name(name: str) -> str:
    """How a bank prints a party name: upper case, no punctuation, at most 30 characters."""
    return re.sub(r"[^A-Z0-9 ]", "", name.upper())[:30].strip()


def transpose_digits(amount: Decimal, rng: random.Random) -> Decimal:
    """Swap two adjacent digits of the rupee part, the classic keying error."""
    rupees, paise = divmod(int(amount * 100), 100)
    digits = list(str(rupees))
    pairs = [i for i in range(len(digits) - 1)
             if digits[i] != digits[i + 1] and not (i == 0 and digits[1] == "0")]
    if not pairs:
        return amount + Decimal("100.00")
    i = rng.choice(pairs)
    digits[i], digits[i + 1] = digits[i + 1], digits[i]
    return (Decimal(int("".join(digits)) * 100 + paise) / 100).quantize(PAISA)


def journal_entry(adjustment: Decimal, other_account: str) -> str:
    """The adjusting entry that moves the book's Bank account by `adjustment`."""
    amount = f"{abs(adjustment):.2f}"
    if adjustment > 0:
        return f"Dr Bank {amount} / Cr {other_account} {amount}"
    return f"Dr {other_account} {amount} / Cr Bank {amount}"


# ---------------------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------------------
class _Builder:
    def __init__(self, config: GeneratorConfig):
        self.cfg = config
        self.rng = random.Random(config.seed)
        year, month = (int(part) for part in config.month.split("-"))
        self.start = date(year, month, 1)
        self.end = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
        self.days = business_days(self.start, self.end)
        self.tag = self.start.strftime("%b%y").upper()                 # e.g. JUN26
        self.month_name = self.start.strftime("%B %Y")                 # e.g. June 2026
        self.prev_month_name = (self.start - timedelta(days=1)).strftime("%B %Y")
        self.events = []
        self.used = set()

    # -- random building blocks ----------------------------------------------------------
    def add(self, event: Event) -> Event:
        event.order = len(self.events)
        self.events.append(event)
        return event

    def digits(self, n: int) -> str:
        return "".join(self.rng.choice("0123456789") for _ in range(n))

    def unique(self, make) -> str:
        while True:
            value = make()
            if value not in self.used:
                self.used.add(value)
                return value

    def utr(self, rtgs: bool = False) -> str:
        return self.unique(lambda: f"{self.rng.choice(BANK_CODES)}{'R' if rtgs else 'N'}{self.digits(11)}")

    def rrn(self) -> str:
        return self.unique(lambda: self.digits(12))

    def cheque_no(self) -> str:
        return self.unique(lambda: self.digits(6))

    def amount(self, median: float, sigma: float = 0.9, low: float = 1500, high: float = 1500000) -> Decimal:
        value = min(max(self.rng.lognormvariate(math.log(median), sigma), low), high)
        if self.rng.random() < 0.4:
            value = round(value)  # many invoices are whole rupees
        return Decimal(str(value)).quantize(PAISA, ROUND_HALF_UP)

    def receipt_mode(self, amount: Decimal) -> str:
        if amount >= 200000 and self.rng.random() < 0.5:
            return "RTGS"  # RTGS minimum is Rs 2 lakh
        options = ["NEFT"] * 45 + ["CHEQUE"] * 25
        if amount <= 500000:
            options += ["IMPS"] * 10
        if amount <= 100000:
            options += ["UPI"] * 20
        return self.rng.choice(options)

    def payment_mode(self, amount: Decimal) -> str:
        if amount >= 200000 and self.rng.random() < 0.5:
            return "RTGS"
        options = ["NEFT"] * 55 + ["CHEQUE"] * 30
        if amount <= 100000:
            options += ["UPI"] * 15
        return self.rng.choice(options)

    # -- ordinary receipts and payments --------------------------------------------------
    def customer_receipt(self, day: date, mode: Optional[str] = None) -> Event:
        amount = self.amount(median=45000)
        mode = mode or self.receipt_mode(amount)
        event = Event("RECEIPT", mode, self.rng.choice(CUSTOMERS), amount, book_date=day,
                      doc=self.unique(lambda: f"INV-{self.rng.randint(1000, 9999)}"))
        if mode == "CHEQUE":
            event.bank_ref = self.cheque_no()
            lag = self.rng.choice([1, 1, 1, 2, 2, 3])   # cheque clearing
        elif mode in ("NEFT", "RTGS"):
            event.bank_ref = self.utr(rtgs=mode == "RTGS")
            lag = self.rng.choice([0, 0, 0, 1]) if mode == "NEFT" else 0
        else:
            event.bank_ref = self.rrn()
            lag = 0
        event.bank_date = add_business_days(day, lag)
        return self.add(event)

    def vendor_payment(self, day: date, mode: Optional[str] = None) -> Event:
        amount = self.amount(median=38000)
        mode = mode or self.payment_mode(amount)
        event = Event("PAYMENT", mode, self.rng.choice(VENDORS), amount, book_date=day,
                      doc=self.unique(lambda: f"BILL-{self.rng.randint(1000, 9999)}"))
        if mode == "CHEQUE":
            event.bank_ref = self.cheque_no()
            lag = self.rng.randint(2, 10)               # vendors take time to present cheques
        elif mode in ("NEFT", "RTGS"):
            event.bank_ref = self.utr(rtgs=mode == "RTGS")
            lag = 0
        else:
            event.bank_ref = self.rrn()
            lag = 0
        event.bank_date = add_business_days(day, lag)
        return self.add(event)

    def ordinary_activity(self) -> None:
        for _ in range(self.cfg.n_receipts):
            self.customer_receipt(self.rng.choice(self.days))
        for _ in range(self.cfg.n_payments):
            self.vendor_payment(self.rng.choice(self.days))

    def batched_cheques(self) -> None:
        """Several cheques paid in on one deposit slip; the bank credits their total."""
        for _ in range(self.cfg.n_batches):
            day = self.rng.choice(self.days[:-4])       # early enough to clear this month
            slip = self.unique(lambda: f"DS{self.digits(6)}")
            lag = self.rng.choice([1, 1, 2])
            for _ in range(self.rng.randint(2, 4)):
                event = self.customer_receipt(day, mode="CHEQUE")
                event.bank_date = add_business_days(day, lag)
                event.batch = slip

    def lookalike_receipts(self) -> None:
        """Receipts of one round amount, on one day, from different customers, with no
        reference. The hardest case for a matcher: only the counterparty name tells them apart."""
        used = set()
        for _ in range(self.cfg.n_lookalike_groups):
            while True:   # a fresh amount and day per group, so no two lines are indistinguishable
                amount = Decimal(self.rng.choice([25000, 50000, 75000, 100000, 118000, 150000])).quantize(PAISA)
                day = self.rng.choice(self.days[:-3])
                if (amount, day) not in used:
                    used.add((amount, day))
                    break
            for customer in self.rng.sample(CUSTOMERS, k=3):
                event = self.customer_receipt(day, mode="NEFT")
                event.counterparty, event.amount, event.bank_date = customer, amount, day
                event.no_reference = True

    def period_end_items(self) -> None:
        """Guarantee some items straddle the month end, as they do in every real close."""
        last = self.days[-1]
        self.customer_receipt(last, mode="CHEQUE")                     # clears next month
        neft = self.customer_receipt(last, mode="NEFT")
        neft.bank_date = add_business_days(last, 1)                    # credited next month
        for day in self.days[-3:-1]:
            cheque = self.vendor_payment(day, mode="CHEQUE")
            cheque.bank_date = add_business_days(day, self.rng.randint(4, 9))  # not yet presented

    def recurring_payments(self) -> None:
        def fixed(counterparty, amount, day, mode, particulars, reference, narration):
            return self.add(Event("PAYMENT", mode, counterparty, amount, book_date=day, bank_date=day,
                                  bank_ref=reference, particulars=particulars, reference=reference,
                                  narration=narration, recurring=True))

        def on_or_before(day_of_month):
            return max(d for d in self.days if d.day <= day_of_month)

        def on_or_after(day_of_month):
            return min(d for d in self.days if d.day >= day_of_month)

        utr = self.utr()
        fixed("Shree Estates", Decimal("350000.00"), self.days[2], "NEFT",
              f"Office rent for {self.month_name} - Shree Estates", utr, f"NEFT DR-{utr}-SHREE ESTATES")

        day = on_or_before(7)
        cin = f"{self.digits(7)}{day:%d%m%y}{self.digits(5)}"   # BSR code + date + challan serial
        fixed("Income Tax Department", self.amount(145000, 0.2, 90000, 220000), day, "NETBANKING",
              f"TDS deposit for {self.prev_month_name} (Challan 281)", cin, f"ITNS281 TDS CIN {cin}")

        umrn = self.unique(lambda: f"{self.rng.choice(BANK_CODES)}{self.digits(16)}")
        fixed("Unnati Finance Ltd", Decimal("62340.00"), on_or_after(5), "NACH",
              "Term loan EMI - Unnati Finance Ltd", umrn, f"NACH DR {umrn} UNNATI FINANCE LTD")

        cpin = self.unique(lambda: self.digits(14))
        fixed("GST (GSTR-3B)", self.amount(610000, 0.2, 350000, 900000), on_or_before(20), "NETBANKING",
              f"GST payment for {self.prev_month_name} (GSTR-3B)", cpin, f"GST CHALLAN CPIN {cpin}")

        utr = self.utr()
        fixed("Employees (payroll)", self.amount(2450000, 0.05, 2000000, 3000000), self.days[-1], "NEFT",
              f"Salaries for {self.month_name}", utr, f"NEFT DR BULK SAL {self.tag} {utr}")

    def bank_only_items(self) -> None:
        charges = BANK_CHARGES + (TRICKY_BANK_CHARGES if self.cfg.tricky_narrations else [])
        debits = UNRECORDED_DEBITS + (TRICKY_UNRECORDED_DEBITS if self.cfg.tricky_narrations else [])
        for label, amount in self.rng.sample(charges, k=min(self.cfg.n_bank_charges, len(charges))):
            day = self.days[-1] if "TXN" in label else self.rng.choice(self.days)
            self.add(Event("PAYMENT", "CHARGE", "Bank", Decimal(amount), bank_date=day,
                           narration=f"{label} {self.tag}", category=BANK_CHARGE,
                           other_account="Bank Charges"))
        if self.cfg.interest_credit:
            self.add(Event("RECEIPT", "INTEREST", "Bank", self.amount(7000, 0.3, 3000, 12000),
                           bank_date=self.days[-1], narration="INT CR SWEEP FD",
                           category=INTEREST_CREDIT, other_account="Interest Income"))
        for payee, account, amount in self.rng.sample(debits, k=min(self.cfg.n_unrecorded_debits, len(debits))):
            umrn = self.unique(lambda: f"{self.rng.choice(BANK_CODES)}{self.digits(16)}")
            self.add(Event("PAYMENT", "NACH", payee, Decimal(amount), bank_date=self.rng.choice(self.days),
                           bank_ref=umrn, narration=f"NACH DR {umrn} {bank_name(payee)}",
                           category=UNRECORDED_DEBIT, other_account=account))
        for _ in range(self.cfg.n_direct_credits):
            customer = self.rng.choice(CUSTOMERS)
            utr = self.utr()
            self.add(Event("RECEIPT", "NEFT", customer, self.amount(60000, 0.6, 10000, 300000),
                           bank_date=self.rng.choice(self.days), bank_ref=utr,
                           narration=f"NEFT CR-{utr}-{self.remitter_name(customer)}-ADVANCE",
                           category=DIRECT_CREDIT, other_account=f"Customer Advances - {customer}"))

    def remitter_name(self, customer: str) -> str:
        """Direct credits often arrive under a shortened remitter name."""
        if self.cfg.tricky_narrations and self.rng.random() < 0.5:
            return self.mangle(customer)
        return bank_name(customer)

    # -- injected errors -----------------------------------------------------------------
    def inject_errors(self) -> None:
        eligible = [e for e in self.events
                    if e.book_date and e.bank_date and e.bank_date <= self.end
                    and not e.batch and not e.recurring and not e.category]
        for event in eligible:
            if event.mode != "CHEQUE" and self.rng.random() < self.cfg.no_reference_rate:
                event.no_reference = True       # cheque numbers are always printed, so never for cheques
        clean = [e for e in eligible if not e.no_reference]
        for event in clean:
            if self.rng.random() < self.cfg.amount_error_rate:
                event.book_amount = transpose_digits(event.amount, self.rng)
        if not any(e.book_amount is not None for e in clean) and clean:
            clean[0].book_amount = transpose_digits(clean[0].amount, self.rng)
        pool = [e for e in clean if e.book_amount is None]
        duplicated = set()
        for original in self.rng.sample(pool, k=min(self.cfg.n_duplicates, len(pool))):
            self.add(dataclasses.replace(original, bank_date=None, category=DUPLICATE_BOOK,
                                         duplicate_of=original))
            duplicated.add(id(original))
        if self.cfg.n_tds:
            self.settle_net_of_tds([e for e in pool if id(e) not in duplicated])

    def settle_net_of_tds(self, pool: list) -> None:
        """Customers deduct TDS before paying, and the company deducts it before paying vendors, but
        the cash book shows the gross amount. Runs after every other random choice and draws from its
        own random stream, so switching it on changes only the settlements it picks."""
        rng = random.Random(self.cfg.seed * 7919 + 1)
        rates = [Decimal("0.001"), Decimal("0.01"), Decimal("0.02"), Decimal("0.02"), Decimal("0.10")]
        for event in rng.sample(pool, k=min(self.cfg.n_tds, len(pool))):
            event.tds_rate, event.tds_before_gst = rng.choice(rates), rng.random() < 0.5
            base = event.amount / Decimal("1.18") if event.tds_before_gst else event.amount
            tds = (base * event.tds_rate).quantize(PAISA, ROUND_HALF_UP)
            if rng.random() < 0.7:
                tds = tds.quantize(Decimal("1"), ROUND_HALF_UP)   # most deductors round TDS to the rupee
            event.book_amount = event.amount             # the book shows the gross invoice or bill
            event.amount = event.amount - tds            # the bank shows what actually moved
            event.category = TDS_DEDUCTED

    def categorize(self) -> None:
        for event in self.events:
            if event.book_amount is None:
                event.book_amount = event.amount if event.book_date else None
            if event.category:
                continue                        # bank-only items and duplicates are already set
            if event.bank_date > self.end:
                event.category = DEPOSIT_IN_TRANSIT if event.kind == "RECEIPT" else OUTSTANDING_PAYMENT
            elif event.batch:
                event.category = BATCH_DEPOSIT
            elif event.book_amount != event.amount:
                event.category = AMOUNT_ERROR
            elif event.no_reference:
                event.category = NO_REFERENCE
            elif event.bank_date > event.book_date:
                event.category = DATE_LAG
            else:
                event.category = EXACT

    # -- rendering -------------------------------------------------------------------------
    def mangle(self, name: str) -> str:
        """A bank's shortened version of a name, with nothing to anchor an exact match."""
        plain = bank_name(name)
        words = plain.split()
        styles = [
            lambda: plain[: self.rng.randint(9, 14)].strip(),
            lambda: plain.replace(" ", "")[:16],
            lambda: " ".join(w[:4] for w in words),
            lambda: " ".join(w[0] for w in words[:-1]) + " " + words[-1],
        ]
        for style in self.rng.sample(styles, k=len(styles)):
            mangled = style()
            if mangled != plain:
                return mangled
        return plain[:-3]

    def book_text(self, event: Event) -> tuple:
        """(particulars, reference) as the company recorded them."""
        if event.recurring:
            return event.particulars, event.reference
        who = event.counterparty
        if event.kind == "RECEIPT":
            if event.mode == "CHEQUE":
                return f"Cheque {event.bank_ref} from {who} against {event.doc}", event.bank_ref
            return f"Received from {who} against {event.doc}", event.doc
        if event.mode == "CHEQUE":
            return f"Cheque {event.bank_ref} issued to {who} for {event.doc}", event.bank_ref
        # For payments the company normally keeps the UTR; without it, only its own bill number.
        return f"Paid to {who} for {event.doc}", event.doc if event.no_reference else event.bank_ref

    def bank_text(self, event: Event) -> str:
        if event.narration:
            return event.narration
        name = self.mangle(event.counterparty) if event.no_reference else bank_name(event.counterparty)
        doc = "" if event.no_reference else f"-{event.doc}"
        ref, mode = event.bank_ref, event.mode
        if event.kind == "RECEIPT":
            if mode == "CHEQUE":
                return f"CLG CHQ {ref} {name[:18]}"
            if mode in ("NEFT", "RTGS"):
                return f"{mode} CR-{ref}-{name}{doc}"
            prefix = "IMPS/P2A" if mode == "IMPS" else "UPI/CR"
            return f"{prefix}/{ref}/{name[:16]}{doc.replace('-', '/', 1)}"
        if mode == "CHEQUE":
            return f"CHQ PAID {ref} {name[:18]}"
        if mode in ("NEFT", "RTGS"):
            return f"{mode} DR-{ref}-{name}"
        return f"UPI/DR/{ref}/{name[:16]}"

    def book_lines(self) -> list:
        booked = sorted((e for e in self.events if e.book_date), key=lambda e: (e.book_date, e.order))
        lines, balance, counters = [], self.cfg.opening_balance, {"RECEIPT": 0, "PAYMENT": 0}
        for number, event in enumerate(booked, start=1):
            counters[event.kind] += 1
            event.book_id = f"BK-{number:04d}"
            event.voucher = f"{'RV' if event.kind == 'RECEIPT' else 'PV'}-{counters[event.kind]:04d}"
            balance += event.book_amount * event.sign
            particulars, reference = self.book_text(event)
            lines.append({
                "book_id": event.book_id, "date": event.book_date, "voucher_no": event.voucher,
                "party": event.counterparty, "particulars": particulars, "mode": event.mode, "reference": reference,
                "receipt": event.book_amount if event.kind == "RECEIPT" else None,
                "payment": event.book_amount if event.kind == "PAYMENT" else None,
                "balance": balance,
            })
        return lines

    def bank_lines(self) -> list:
        postings, slips = [], {}
        for event in self.events:
            if event.bank_date is None or event.bank_date > self.end:
                continue
            if event.batch:
                slips.setdefault(event.batch, []).append(event)
            else:
                postings.append([event])
        postings += list(slips.values())
        postings.sort(key=lambda members: (members[0].bank_date, members[0].order))
        lines, balance = [], self.cfg.opening_balance
        for number, members in enumerate(postings, start=1):
            first = members[0]
            amount = sum((m.amount for m in members), ZERO)
            for member in members:
                member.bank_id = f"BS-{number:04d}"
            if first.batch:
                narration = f"CLG DEP {len(members)} CHQS SLIP {first.batch}"
                bank_ref = first.batch
            else:
                narration, bank_ref = self.bank_text(first), first.bank_ref
            balance += amount * first.sign
            lines.append({
                "bank_id": first.bank_id, "value_date": first.bank_date, "narration": narration,
                "bank_ref": bank_ref,
                "deposit": amount if first.kind == "RECEIPT" else None,
                "withdrawal": amount if first.kind == "PAYMENT" else None,
                "balance": balance,
            })
        return lines

    def answer_key(self) -> list:
        groups, seen = [], set()
        for event in sorted(self.events, key=lambda e: (e.book_date or e.bank_date, e.order)):
            if event.batch:
                if event.batch not in seen:
                    seen.add(event.batch)
                    groups.append([e for e in self.events if e.batch == event.batch])
            else:
                groups.append([event])

        items = []
        for number, members in enumerate(groups, start=1):
            first = members[0]
            category = first.category
            book_amount = sum((m.book_amount * m.sign for m in members if m.book_id), ZERO)
            bank_amount = sum((m.amount * m.sign for m in members if m.bank_id), ZERO)
            adjusts, adjustment, entry, clears = "NONE", ZERO, "No entry needed", None
            if category == AMOUNT_ERROR:
                adjusts, adjustment = "BOOK", bank_amount - book_amount
                entry = journal_entry(adjustment, self.ledger_account(first))
            elif category == TDS_DEDUCTED:
                adjusts, adjustment = "BOOK", bank_amount - book_amount
                entry = journal_entry(adjustment, "TDS Receivable" if first.kind == "RECEIPT" else "TDS Payable")
            elif category == DUPLICATE_BOOK:
                adjusts, adjustment = "BOOK", -book_amount
                entry = journal_entry(adjustment, self.ledger_account(first))
            elif category in (BANK_CHARGE, INTEREST_CREDIT, UNRECORDED_DEBIT, DIRECT_CREDIT):
                adjusts, adjustment = "BOOK", bank_amount
                entry = journal_entry(adjustment, first.other_account)
            elif category in BANK_SIDE:
                adjusts, adjustment = "BANK", first.amount * first.sign
                entry = "No entry needed: timing difference"
                clears = first.bank_date
            items.append({
                "item_id": f"AK-{number:03d}", "category": category,
                "book_ids": [m.book_id for m in members if m.book_id],
                "bank_ids": sorted({m.bank_id for m in members if m.bank_id}),
                "book_amount": book_amount if any(m.book_id for m in members) else None,
                "bank_amount": bank_amount if any(m.bank_id for m in members) else None,
                "adjusts": adjusts, "adjustment": adjustment, "suggested_entry": entry,
                "expected_clear_date": clears, "note": self.note(members),
            })
        return items

    @staticmethod
    def ledger_account(event: Event) -> str:
        prefix = "Accounts Receivable" if event.kind == "RECEIPT" else "Accounts Payable"
        return f"{prefix} - {event.counterparty}"

    def note(self, members: list) -> str:
        e = members[0]
        if e.category == EXACT:
            return "Same amount, same day and a shared reference."
        if e.category == DATE_LAG:
            lag = len(business_days(e.book_date + timedelta(days=1), e.bank_date))
            return f"Same amount and reference; the bank posted it {lag} working day(s) later."
        if e.category == NO_REFERENCE:
            return "Same amount; the bank narration has no shared reference and a shortened name."
        if e.category == BATCH_DEPOSIT:
            return f"{len(members)} cheques paid in on slip {e.batch}; the bank shows one combined credit."
        if e.category == AMOUNT_ERROR:
            return f"Keyed in as {e.book_amount:.2f} instead of {e.amount:.2f} (two digits transposed)."
        if e.category == TDS_DEDUCTED:
            base = "the value before 18% GST" if e.tds_before_gst else "the full amount"
            return (f"Settled net of TDS at {(e.tds_rate * 100).normalize():f}% on {base}: the bank shows {e.amount:.2f}, "
                    f"the book the gross {e.book_amount:.2f}.")
        if e.category == DUPLICATE_BOOK:
            original = e.duplicate_of
            return f"Duplicate of {original.voucher} ({original.book_id}); it happened only once."
        if e.category == DEPOSIT_IN_TRANSIT:
            return f"Booked on {e.book_date}; the bank credits it on {e.bank_date}, after this statement."
        if e.category == OUTSTANDING_PAYMENT:
            return f"Cheque {e.bank_ref} issued on {e.book_date} has not been presented yet."
        if e.category == BANK_CHARGE:
            return "Charge debited by the bank and not yet booked."
        if e.category == INTEREST_CREDIT:
            return "Interest credited by the bank and not yet booked."
        if e.category == UNRECORDED_DEBIT:
            return f"NACH auto-debit to {e.counterparty} that was never booked."
        return f"{e.counterparty} paid straight into the account; not yet booked."

    def build(self) -> GeneratedData:
        self.ordinary_activity()
        self.batched_cheques()
        self.lookalike_receipts()
        self.period_end_items()
        self.recurring_payments()
        self.bank_only_items()
        self.inject_errors()
        self.categorize()
        book, bank = self.book_lines(), self.bank_lines()
        items = self.answer_key()

        opening = self.cfg.opening_balance
        book_closing = book[-1]["balance"] if book else opening
        bank_closing = bank[-1]["balance"] if bank else opening
        book_adjustments = sum((i["adjustment"] for i in items if i["adjusts"] == "BOOK"), ZERO)
        bank_adjustments = sum((i["adjustment"] for i in items if i["adjusts"] == "BANK"), ZERO)
        adjusted_book = book_closing + book_adjustments
        adjusted_bank = bank_closing + bank_adjustments
        difference = adjusted_bank - adjusted_book
        if difference != ZERO:
            raise AssertionError(f"Generated data does not reconcile: difference {difference}")

        categories = CATEGORIES + (OPTIONAL_CATEGORIES if self.cfg.n_tds else [])
        counts = {c: sum(1 for i in items if i["category"] == c) for c in categories}
        summary = {
            "period": self.cfg.month, "seed": self.cfg.seed, "company": COMPANY,
            "opening_balance": f"{opening:.2f}",
            "book_closing_balance": f"{book_closing:.2f}",
            "bank_closing_balance": f"{bank_closing:.2f}",
            "book_adjustments": f"{book_adjustments:.2f}",
            "bank_adjustments": f"{bank_adjustments:.2f}",
            "adjusted_book_balance": f"{adjusted_book:.2f}",
            "adjusted_bank_balance": f"{adjusted_bank:.2f}",
            "difference": f"{difference:.2f}",
            "book_lines": len(book), "bank_lines": len(bank), "items": len(items),
            "items_by_category": counts,
        }
        return GeneratedData(self.cfg, book, bank, items, summary)


def generate(config: Optional[GeneratorConfig] = None) -> GeneratedData:
    """Build one month of cash book, bank statement and answer key. Same seed, same data."""
    return _Builder(config or GeneratorConfig()).build()


# ---------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------
BOOK_COLUMNS = ["book_id", "date", "voucher_no", "party", "particulars", "mode", "reference",
                "receipt", "payment", "balance"]
BANK_COLUMNS = ["bank_id", "value_date", "narration", "bank_ref", "deposit", "withdrawal", "balance"]
KEY_COLUMNS = ["item_id", "category", "book_ids", "bank_ids", "book_amount", "bank_amount",
               "adjusts", "adjustment", "suggested_entry", "expected_clear_date", "note"]


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return ";".join(value)
    return str(value)


def _write_csv(path: Path, columns: list, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_fmt(row[column]) for column in columns])


def write_outputs(data: GeneratedData, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "cash_book.csv", BOOK_COLUMNS, data.book)
    _write_csv(out_dir / "bank_statement.csv", BANK_COLUMNS, data.bank)
    _write_csv(out_dir / "answer_key.csv", KEY_COLUMNS, data.answer_key)
    _write_csv(out_dir / "parties.csv", ["party", "type"], party_master())
    with (out_dir / "summary.json").open("w", newline="\n", encoding="utf-8") as handle:
        json.dump(data.summary, handle, indent=2)
        handle.write("\n")


def party_master() -> list:
    """The company's customer and vendor master list (what an ERP would hold)."""
    return ([{"party": name, "type": "CUSTOMER"} for name in CUSTOMERS]
            + [{"party": name, "type": "VENDOR"} for name in VENDORS])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate a cash book, bank statement and answer key.")
    parser.add_argument("--month", default="2026-06", help="statement period, YYYY-MM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data", help="output folder")
    parser.add_argument("--tds", type=int, default=0, help="receipts and payments settled net of TDS (default 0)")
    args = parser.parse_args(argv)

    data = generate(GeneratorConfig(month=args.month, seed=args.seed, n_tds=args.tds))
    write_outputs(data, args.out)
    s = data.summary
    print(f"Generated {args.month} (seed {args.seed}) into {args.out}/")
    print(f"  cash book       {s['book_lines']:>4} lines   closing balance {Decimal(s['book_closing_balance']):>15,.2f}")
    print(f"  bank statement  {s['bank_lines']:>4} lines   closing balance {Decimal(s['bank_closing_balance']):>15,.2f}")
    print(f"  answer key      {s['items']:>4} items")
    for category, count in s["items_by_category"].items():
        print(f"      {category:<20} {count:>4}")
    print(f"  adjusted book {Decimal(s['adjusted_book_balance']):,.2f} = adjusted bank "
          f"{Decimal(s['adjusted_bank_balance']):,.2f} (difference {s['difference']})")


if __name__ == "__main__":
    main()
