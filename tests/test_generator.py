"""Tests for the statement generator: determinism, completeness and the reconciliation tie-out."""
from datetime import date, timedelta
from decimal import Decimal

import pytest

from recon import generator as g

PERIOD_END = date(2026, 6, 30)  # the default config's month


@pytest.fixture(scope="module")
def data():
    return g.generate(g.GeneratorConfig())


@pytest.fixture(scope="module")
def by_id(data):
    return {line["book_id"]: line for line in data.book}, {line["bank_id"]: line for line in data.bank}


def signed(row, money_in, money_out):
    return row[money_in] if row[money_in] is not None else -row[money_out]


def shares_reference(book_line, bank_line):
    reference = book_line["reference"]
    return reference == bank_line["bank_ref"] or reference in bank_line["narration"]


def test_same_seed_writes_identical_files(tmp_path):
    for folder in ("a", "b"):
        g.write_outputs(g.generate(g.GeneratorConfig(seed=7)), tmp_path / folder)
    for name in ("cash_book.csv", "bank_statement.csv", "answer_key.csv", "summary.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_different_seeds_give_different_statements():
    assert g.generate(g.GeneratorConfig(seed=1)).bank != g.generate(g.GeneratorConfig(seed=2)).bank


def test_every_line_is_in_exactly_one_answer_key_item(data):
    book_ids = [i for item in data.answer_key for i in item["book_ids"]]
    bank_ids = [i for item in data.answer_key for i in item["bank_ids"]]
    assert sorted(book_ids) == sorted(line["book_id"] for line in data.book)
    assert sorted(bank_ids) == sorted(line["bank_id"] for line in data.bank)


def test_reconciliation_ties_out_to_the_paisa(data):
    book_adjustments = sum((i["adjustment"] for i in data.answer_key if i["adjusts"] == "BOOK"), Decimal(0))
    bank_adjustments = sum((i["adjustment"] for i in data.answer_key if i["adjusts"] == "BANK"), Decimal(0))
    assert data.book[-1]["balance"] + book_adjustments == data.bank[-1]["balance"] + bank_adjustments
    assert data.summary["difference"] == "0.00"


@pytest.mark.parametrize("seed", range(25))
def test_ties_out_for_many_seeds(seed):
    assert g.generate(g.GeneratorConfig(seed=seed)).summary["difference"] == "0.00"


def test_running_balances_and_amounts(data):
    for rows, money_in, money_out in ((data.book, "receipt", "payment"), (data.bank, "deposit", "withdrawal")):
        balance = data.config.opening_balance
        for row in rows:
            amount = row[money_in] if row[money_in] is not None else row[money_out]
            assert amount > 0 and amount == amount.quantize(Decimal("0.01"))
            balance += signed(row, money_in, money_out)
            assert row["balance"] == balance


def test_every_category_appears_with_the_default_settings(data):
    assert {item["category"] for item in data.answer_key} == set(g.CATEGORIES)


def test_matched_items_agree(data, by_id):
    book, bank = by_id
    for item in data.answer_key:
        if item["category"] not in g.MATCHED:
            continue
        assert item["book_amount"] == item["bank_amount"]
        assert item["adjusts"] == "NONE" and item["adjustment"] == 0
        bank_date = bank[item["bank_ids"][0]]["value_date"]
        for book_id in item["book_ids"]:
            assert 0 <= (bank_date - book[book_id]["date"]).days <= 16   # cheques can take 10 working days
        if item["category"] == g.EXACT:
            assert bank_date == book[item["book_ids"][0]]["date"]
        if item["category"] == g.BATCH_DEPOSIT:
            assert len(item["book_ids"]) >= 2 and len(item["bank_ids"]) == 1


def test_references_are_shared_only_where_expected(data, by_id):
    book, bank = by_id
    for item in data.answer_key:
        if item["category"] in (g.EXACT, g.DATE_LAG, g.AMOUNT_ERROR):
            assert shares_reference(book[item["book_ids"][0]], bank[item["bank_ids"][0]]), item["item_id"]
        if item["category"] == g.NO_REFERENCE:
            assert not shares_reference(book[item["book_ids"][0]], bank[item["bank_ids"][0]]), item["item_id"]


def test_amount_errors_are_digit_transpositions(data):
    for item in data.answer_key:
        if item["category"] == g.AMOUNT_ERROR:
            recorded, actual = f"{abs(item['book_amount']):.2f}", f"{abs(item['bank_amount']):.2f}"
            assert recorded != actual
            assert sorted(recorded) == sorted(actual) or abs(item["book_amount"] - item["bank_amount"]) == 100
            assert item["adjustment"] == item["bank_amount"] - item["book_amount"]


def test_one_sided_items(data):
    for item in data.answer_key:
        category = item["category"]
        if category in (g.DUPLICATE_BOOK, g.DEPOSIT_IN_TRANSIT, g.OUTSTANDING_PAYMENT):
            assert item["book_ids"] and not item["bank_ids"]
        if category in (g.BANK_CHARGE, g.INTEREST_CREDIT, g.UNRECORDED_DEBIT, g.DIRECT_CREDIT):
            assert item["bank_ids"] and not item["book_ids"]
            assert item["adjusts"] == "BOOK" and item["adjustment"] == item["bank_amount"]


def test_in_transit_items_clear_after_the_period(data):
    for item in data.answer_key:
        if item["category"] in g.BANK_SIDE:
            assert item["adjusts"] == "BANK"
            assert item["expected_clear_date"] > PERIOD_END


def test_duplicates_have_a_twin_with_the_same_reference(data, by_id):
    book, _ = by_id
    for item in data.answer_key:
        if item["category"] == g.DUPLICATE_BOOK:
            duplicate = book[item["book_ids"][0]]
            twins = [line for line in data.book
                     if line["reference"] == duplicate["reference"] and line["book_id"] != duplicate["book_id"]]
            assert len(twins) == 1
            assert (twins[0]["receipt"], twins[0]["payment"]) == (duplicate["receipt"], duplicate["payment"])


def test_suggested_entries_balance(data):
    for item in data.answer_key:
        if item["adjusts"] == "BOOK":
            entry, amount = item["suggested_entry"], f"{abs(item['adjustment']):.2f}"
            assert entry.count(amount) == 2                      # same amount on the debit and credit side
            assert entry.startswith(f"Dr Bank {amount} /") == (item["adjustment"] > 0)  # money into Bank


def test_recurring_payments_are_booked_and_matched(data, by_id):
    book, _ = by_id
    recurring = {"Salaries for June 2026", "Office rent for June 2026 - Shree Estates",
                 "Term loan EMI - Unnati Finance Ltd"}
    matched = {book[i]["particulars"] for item in data.answer_key if item["category"] == g.EXACT
               for i in item["book_ids"]}
    assert recurring <= matched
