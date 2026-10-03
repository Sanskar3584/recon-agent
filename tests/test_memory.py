"""Payee memory tests: keys, storage, the labeler wrapper, and the month-to-month evaluation."""
import json
from datetime import date
from decimal import Decimal

import pytest

from recon import evaluate, generator, pipeline
from recon.agent import BANK_ONLY, NEEDS_REVIEW, Label, RulesLabeler, label_exceptions, resolve
from recon.loader import BankLine
from recon.memory import MemoryLabeler, PayeeMemory, payee_key
from recon.score import account_in_entry


def bank(narration, amount="-12650.00", bank_id="BS-0001"):
    return BankLine(bank_id, date(2026, 6, 15), narration, "", Decimal(amount))


def test_key_keeps_the_payee_and_drops_references_and_dates():
    june = bank("NACH DR QRTB1234567890123456 AAROGYA HEALTH PLAN")
    july = bank("NACH DR ZENB9876543210987654 AAROGYA HEALTH PLAN")
    assert payee_key(june) == payee_key(july) == "OUT NACH DR AAROGYA HEALTH PLAN"
    assert payee_key(bank("LOCKER RENT FY26-27 JUN26")) == payee_key(bank("LOCKER RENT FY26-27 JUL26"))


def test_direction_is_part_of_the_key():
    assert payee_key(bank("SHAKTI LOGISTICS", "-100.00")) != payee_key(bank("SHAKTI LOGISTICS", "100.00"))


def test_vague_narrations_are_not_remembered():
    memory = PayeeMemory()
    line = bank("NEFT CR-QRTBN12345678901", "5000.00")
    assert payee_key(line) is None
    assert not memory.remember(line, "DIRECT_CREDIT", "Customer Advances - X", "2026-06-30")


def test_only_bank_only_categories_are_remembered():
    assert not PayeeMemory().remember(bank("NACH DR X1 SURAKSHA LIFE"), "AMOUNT_ERROR", "", "2026-06-30")


def test_remember_recall_and_persist(tmp_path):
    path = tmp_path / "memory.json"
    memory = PayeeMemory(path)
    line = bank("NACH DR QRTB1 AAROGYA HEALTH PLAN")
    assert memory.recall(line) is None
    assert memory.remember(line, "UNRECORDED_DEBIT", "Insurance Expense", "2026-05-31")
    assert memory.remember(line, "UNRECORDED_DEBIT", "Insurance Expense", "2026-06-30")
    memory.save()
    again = PayeeMemory(path).recall(bank("NACH DR ZENB2 AAROGYA HEALTH PLAN"))
    assert (again.category, again.counter_account, again.approved_on, again.approvals) == \
        ("UNRECORDED_DEBIT", "Insurance Expense", "2026-06-30", 2)
    assert json.loads(path.read_text())["OUT NACH DR AAROGYA HEALTH PLAN"]["approvals"] == 2


def test_the_latest_approval_wins():
    memory, line = PayeeMemory(), bank("NACH DR QRTB1 AAROGYA HEALTH PLAN")
    memory.remember(line, "UNRECORDED_DEBIT", "Miscellaneous Expense", "2026-05-31")
    memory.remember(line, "UNRECORDED_DEBIT", "Insurance Expense", "2026-06-30")
    hit = memory.recall(line)
    assert hit.counter_account == "Insurance Expense" and hit.approvals == 1


@pytest.fixture(scope="module")
def month(tmp_path_factory):
    folder = tmp_path_factory.mktemp("month")
    generator.write_outputs(generator.generate(generator.GeneratorConfig(seed=42)), folder)
    return pipeline.reconcile(folder / "cash_book.csv", folder / "bank_statement.csv", folder / "parties.csv",
                              RulesLabeler())


def test_known_payees_never_reach_the_wrapped_labeler(month):
    debit = next(c for c in month.cases if c.kind == BANK_ONLY and "NACH" in c.bank[0].narration)
    memory = PayeeMemory()
    memory.remember(debit.bank[0], "UNRECORDED_DEBIT", "Lease Rental Expense", "2026-05-31")
    sent = []

    class Inner:
        name = "inner"

        def label_all(self, cases):
            sent.extend(cases)
            return RulesLabeler().label_all(cases)

    labeler = MemoryLabeler(Inner(), memory)
    suggestions = label_exceptions(month.cases, labeler)
    assert labeler.name == "inner+memory" and labeler.hits == 1
    assert debit.exception_id not in {c.exception_id for c in sent} and len(sent) == len(month.cases) - 1
    s = next(s for s in suggestions if s.exception_id == debit.exception_id)
    assert s.journal_entry.startswith("Dr Lease Rental Expense ") and "approved on 2026-05-31" in s.explanation


def test_remembered_labels_still_pass_the_guardrails(month):
    debit = next(c for c in month.cases if c.kind == BANK_ONLY and "NACH" in c.bank[0].narration)
    memory = PayeeMemory()
    memory.remember(debit.bank[0], "UNRECORDED_DEBIT", "Party Expense", "2026-05-31")   # not in the chart
    label = MemoryLabeler(RulesLabeler(), memory).label_all([debit])[0]
    assert resolve(debit, label, "memory").category == NEEDS_REVIEW


def test_reading_the_account_from_an_entry():
    assert account_in_entry("Dr Insurance Expense 12650.00 / Cr Bank 12650.00") == "Insurance Expense"
    assert account_in_entry("Dr Bank 5000.00 / Cr Customer Advances - Meera Foods Pvt Ltd 5000.00") == \
        "Customer Advances - Meera Foods Pvt Ltd"
    assert account_in_entry("Dr Bank Charges 236.00 / Cr Bank 236.00") == "Bank Charges"


def test_memory_learns_across_months(tmp_path):
    months = evaluate.months_ending("2026-06", 6)
    plain = evaluate.evaluate(months, RulesLabeler(), tmp_path / "a")["summary"]
    learned = evaluate.evaluate(months, RulesLabeler(), tmp_path / "b", memory=PayeeMemory())["summary"]
    assert learned["labeler"] == "rules+memory" and learned["answered_from_memory"] > 0
    assert learned["entry_accuracy_pct"] > plain["entry_accuracy_pct"]
    assert learned["needs_review"] < plain["needs_review"]
    debit = lambda s: s["by_category"]["UNRECORDED_DEBIT"]["entry_correct"]
    assert debit(learned) > debit(plain)


def test_pipeline_command_line_reports_memory_hits(tmp_path, capsys):
    folder = tmp_path / "data"
    generator.write_outputs(generator.generate(generator.GeneratorConfig(seed=7)), folder)
    pipeline.main(["--data", str(folder), "--out", str(tmp_path / "out"), "--memory", str(tmp_path / "m.json")])
    assert "Answered from payee memory: 0 line(s)" in capsys.readouterr().out
