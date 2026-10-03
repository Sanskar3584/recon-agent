"""Segment 1 tests: the loader reads both files exactly, and rejects files it can't trust."""
from decimal import Decimal

import pytest

from recon import generator
from recon.loader import LoadError, load_bank, load_book


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("data")
    data = generator.generate(generator.GeneratorConfig(seed=42))
    generator.write_outputs(data, folder)
    return data, folder


def test_loads_every_line_with_signed_amounts(files):
    data, folder = files
    book, bank = load_book(folder / "cash_book.csv"), load_bank(folder / "bank_statement.csv")
    assert [line.book_id for line in book] == [row["book_id"] for row in data.book]
    assert [line.bank_id for line in bank] == [row["bank_id"] for row in data.bank]
    for line, row in zip(book, data.book):
        assert line.amount == (row["receipt"] if row["receipt"] is not None else -row["payment"])
        assert line.date == row["date"] and line.party == row["party"]
    for line, row in zip(bank, data.bank):
        assert line.amount == (row["deposit"] if row["deposit"] is not None else -row["withdrawal"])


def test_balances_rebuild_from_the_loaded_amounts(files):
    data, folder = files
    opening = data.config.opening_balance
    assert opening + sum(l.amount for l in load_book(folder / "cash_book.csv")) == data.book[-1]["balance"]
    assert opening + sum(l.amount for l in load_bank(folder / "bank_statement.csv")) == data.bank[-1]["balance"]


def write(tmp_path, text, name="bank.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


HEADER = "bank_id,value_date,narration,bank_ref,deposit,withdrawal\n"


def test_keeps_leading_zeros_and_reads_excel_style_amounts(tmp_path):
    path = tmp_path / "bank.csv"
    path.write_bytes(("﻿" + HEADER + 'BS-1,2026-06-02,CLG CHQ 057202 X,057202,"1,25,000.50",\n').encode("utf-8"))
    [line] = load_bank(path)
    assert line.bank_ref == "057202" and line.amount == Decimal("125000.50")


@pytest.mark.parametrize("row, message", [
    ("BS-1,2026-06-02,X,,100.00,50.00\n", "exactly one"),
    ("BS-1,2026-06-02,X,,,\n", "exactly one"),
    ("BS-1,02/06/2026,X,,100.00,\n", "date"),
    ("BS-1,2026-06-02,X,,abc,\n", "not an amount"),
    ("BS-1,2026-06-02,X,,100.00,\nBS-1,2026-06-03,Y,,5.00,\n", "repeated ids"),
])
def test_rejects_bad_rows(tmp_path, row, message):
    with pytest.raises(LoadError, match=message):
        load_bank(write(tmp_path, HEADER + row))


def test_rejects_missing_columns(tmp_path):
    with pytest.raises(LoadError, match="missing columns"):
        load_book(write(tmp_path, "book_id,date\nBK-1,2026-06-01\n", "book.csv"))
