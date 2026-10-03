"""Segment 2 tests: each pass on small hand-made cases, then invariants on generated data."""
from datetime import date
from decimal import Decimal

import pytest

from recon import generator
from recon.loader import BankLine, BookLine, load_bank, load_book
from recon.matcher import MATCHED, NEEDS_CORRECTION, match, shares_reference


def book(book_id, day, amount, reference="", party="Arjun Textiles Pvt Ltd", mode="NEFT"):
    return BookLine(book_id, date(2026, 6, day), f"V-{book_id}", party, "text", mode, reference, Decimal(amount))


def bank(bank_id, day, amount, narration, bank_ref=""):
    return BankLine(bank_id, date(2026, 6, day), narration, bank_ref, Decimal(amount))


def only(result):
    assert len(result.matches) == 1
    return result.matches[0]


def test_pass_1_shared_reference_and_amount():
    m = only(match([book("B1", 1, "1000.00", "INV-1001")],
                   [bank("K1", 2, "1000.00", "NEFT CR-ZENBN1-ARJUN TEXTILES-INV-1001", "ZENBN1")]))
    assert (m.rule, m.status, m.book_ids, m.bank_ids) == ("reference", MATCHED, ["B1"], ["K1"])


def test_pass_2_shared_reference_different_amount_needs_correction():
    m = only(match([book("B1", 1, "5430.00", "INV-2002")],
                   [bank("K1", 1, "5340.00", "NEFT CR-X-ARJUN TEXTILES-INV-2002")]))
    assert (m.rule, m.status) == ("amount_diff", NEEDS_CORRECTION)
    assert m.difference == Decimal("-90.00")


def test_pass_3_same_amount_close_dates_without_reference():
    m = only(match([book("B1", 1, "-56576.71", "BILL-2667", party="Jyoti Office Supplies")],
                   [bank("K1", 1, "-56576.71", "NEFT DR-PLMBN255-J O SUPPLIES", "PLMBN255")]))
    assert (m.rule, m.status) == ("amount_date", MATCHED)


def test_pass_3_name_similarity_breaks_ties():
    result = match([book("B1", 3, "-2000.00", "BILL-1", party="Shakti Logistics")],
                   [bank("K1", 3, "-2000.00", "NEFT DR-A-PRAGATI PRINTERS"),
                    bank("K2", 3, "-2000.00", "NEFT DR-B-SHAKTI LOGIS")])
    assert result.matches[0].bank_ids == ["K2"] and [l.bank_id for l in result.unmatched_bank] == ["K1"]


def test_pass_4_one_bank_credit_for_several_cheques():
    books = [book("B1", 2, "100.00", "111111", mode="CHEQUE"), book("B2", 2, "200.00", "222222", mode="CHEQUE"),
             book("B3", 2, "300.50", "333333", mode="CHEQUE"), book("B4", 2, "999.00", "444444", mode="CHEQUE")]
    result = match(books, [bank("K1", 3, "600.50", "CLG DEP 3 CHQS SLIP DS1", "DS1")])
    m = only(result)
    assert (m.rule, sorted(m.book_ids)) == ("one_to_many", ["B1", "B2", "B3"])
    assert [l.book_id for l in result.unmatched_book] == ["B4"]


def test_a_duplicate_book_entry_is_left_over():
    books = [book("B1", 5, "28525.00", "INV-3902"), book("B2", 5, "28525.00", "INV-3902")]
    result = match(books, [bank("K1", 5, "28525.00", "NEFT CR-X-SAHYADRI-INV-3902")])
    assert only(result).book_ids == ["B1"] and [l.book_id for l in result.unmatched_book] == ["B2"]


def test_direction_must_agree():
    result = match([book("B1", 1, "500.00", "INV-5555")], [bank("K1", 1, "-500.00", "NEFT DR-INV-5555")])
    assert result.matches == [] and len(result.unmatched_book) == len(result.unmatched_bank) == 1


def test_postings_far_outside_the_window_are_not_matched():
    result = match([book("B1", 1, "700.00", "INV-7777")], [bank("K1", 30, "700.00", "NEFT CR-INV-7777")])
    assert result.matches == []


def test_short_or_partial_references_are_not_evidence():
    assert not shares_reference(book("B1", 1, "1.00", "12"), bank("K1", 1, "1.00", "NEFT 12 X"))
    assert not shares_reference(book("B1", 1, "1.00", "INV-100"), bank("K1", 1, "1.00", "NEFT INV-1001"))
    assert shares_reference(book("B1", 1, "1.00", "057202"), bank("K1", 1, "1.00", "CLG CHQ 057202 X"))


@pytest.mark.parametrize("seed", [42, 7, 2024])
def test_every_line_ends_up_in_exactly_one_place(tmp_path, seed):
    generator.write_outputs(generator.generate(generator.GeneratorConfig(seed=seed)), tmp_path)
    books, banks = load_book(tmp_path / "cash_book.csv"), load_bank(tmp_path / "bank_statement.csv")
    result = match(books, banks)
    used_book = [i for m in result.matches for i in m.book_ids] + [l.book_id for l in result.unmatched_book]
    used_bank = [i for m in result.matches for i in m.bank_ids] + [l.bank_id for l in result.unmatched_bank]
    assert sorted(used_book) == sorted(l.book_id for l in books)
    assert sorted(used_bank) == sorted(l.bank_id for l in banks)
    for m in result.matches:
        if m.status == MATCHED:
            assert m.book_amount == m.bank_amount
    assert result.period_end == date(2026, 6, 30)
