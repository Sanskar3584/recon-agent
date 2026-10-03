"""TDS tests: the arithmetic, the labels and entries it leads to, and the opt-in generator scenario."""
import csv
import shutil
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from recon import evaluate, generator, pipeline
from recon.agent import NEEDS_REVIEW, PAIR, Label, RulesLabeler, label_exceptions, resolve
from recon.score import score_labels
from recon.tds import is_transposition, tds_fit

D = Decimal
SAMPLE = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.parametrize("rate", ["0.001", "0.01", "0.02", "0.10"])
@pytest.mark.parametrize("before_gst", [False, True])
@pytest.mark.parametrize("to_the_rupee", [False, True])
def test_fits_every_common_rate_and_base(rate, before_gst, to_the_rupee):
    gross = D("88496.60")
    raw = (gross / D("1.18") if before_gst else gross) * D(rate)
    tds = raw.quantize(D("1") if to_the_rupee else D("0.01"), ROUND_HALF_UP)
    fit = tds_fit(gross, gross - tds)
    assert fit is not None and fit.shortfall == tds and fit.rate == D(rate) and fit.before_gst == before_gst
    assert fit.expected == tds.quantize(D("0.01"))


def test_signs_are_ignored_so_payments_work_too():
    assert tds_fit(D("-50000.00"), D("-49000.00")).rate == D("0.02")


def test_only_a_shortfall_can_be_tds():
    assert tds_fit(D("50000.00"), D("51000.00")) is None
    assert tds_fit(D("50000.00"), D("50000.00")) is None


def test_near_misses_do_not_fit():
    assert tds_fit(D("50000.00"), D("49001.00")) is None      # 999, not 2% (1,000) or 1% (500)
    assert tds_fit(D("50000.00"), D("49000.40")) is None      # 999.60 is neither exact nor rounded


def test_transpositions():
    assert is_transposition(D("45120.50"), D("45210.50"))
    assert is_transposition(D("-21264.00"), D("-21246.00"))
    assert not is_transposition(D("45120.50"), D("45120.50"))
    assert not is_transposition(D("45120.50"), D("45021.50"))   # not neighbours
    assert not is_transposition(D("4512.00"), D("45120.00"))     # different length


def test_the_case_that_used_to_bill_the_customer_for_tds(tmp_path):
    """A customer pays an invoice net of 2% TDS. This used to be called a keying error and booked
    back to the customer's receivable; now it is TDS and goes to TDS Receivable."""
    folder = tmp_path / "month"
    shutil.copytree(SAMPLE, folder)
    path = folder / "bank_statement.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    target = next(r for r in rows if r["narration"].startswith("NEFT CR-") and "-INV-" in r["narration"])
    gross = D(target["deposit"])
    target["deposit"] = f"{gross - (gross * D('0.02')).quantize(D('0.01')):.2f}"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    run = pipeline.reconcile(folder / "cash_book.csv", path, folder / "parties.csv", RulesLabeler())
    s = next(s for s in run.suggestions if target["bank_id"] in s.bank_ids)
    tds = f"{gross - D(target['deposit']):.2f}"
    assert s.category == "TDS_DEDUCTED" and s.journal_entry == f"Dr TDS Receivable {tds} / Cr Bank {tds}"
    assert "owe nothing more" in s.explanation and "Accounts Receivable" not in s.journal_entry


def test_ambiguous_shortfall_goes_to_review(tmp_path):
    """In seed 42's month with plain narrations, a vendor payment shows 21,246.00 at the bank and
    21,264.00 in the book. That is two swapped digits, and also exactly 0.1% TDS on the value before
    18% GST, rounded to the rupee. With two explanations that fit, the rules ask instead of guessing."""
    generator.write_outputs(generator.generate(generator.GeneratorConfig(seed=42, tricky_narrations=False)), tmp_path)
    run = pipeline.reconcile(tmp_path / "cash_book.csv", tmp_path / "bank_statement.csv", tmp_path / "parties.csv",
                             RulesLabeler())
    review = [s for s in run.suggestions if s.kind == PAIR and s.category == NEEDS_REVIEW]
    assert len(review) == 1 and "Two explanations fit" in review[0].explanation
    assert review[0].adjustment == 0 and abs(review[0].amount) == D("21246.00")


@pytest.fixture(scope="module")
def tds_months(tmp_path_factory):
    months = []
    for seed in range(6):
        folder = tmp_path_factory.mktemp(f"tds{seed}")
        generator.write_outputs(generator.generate(generator.GeneratorConfig(seed=seed, n_tds=5)), folder)
        run = pipeline.reconcile(folder / "cash_book.csv", folder / "bank_statement.csv", folder / "parties.csv",
                                 RulesLabeler(), folder / "answer_key.csv")
        months.append(run)
    return months


def test_generated_tds_months_tie_out_and_match_perfectly(tds_months):
    for run in tds_months:
        assert run.scores["matching"]["match_precision_pct"] == 100.0
        assert run.scores["matching"]["exceptions_surfaced_pct"] == 100.0
        assert run.scores["labels"]["by_category"]["TDS_DEDUCTED"]["items"] == 5


def test_rules_label_generated_tds_right_or_send_it_to_review(tds_months):
    for run in tds_months:
        stats = run.scores["labels"]["by_category"]["TDS_DEDUCTED"]
        review = [s for s in run.suggestions if s.kind == PAIR and s.category == NEEDS_REVIEW]
        assert stats["entry_correct"] + len(review) >= stats["items"]
        for s in run.suggestions:
            if s.category == "TDS_DEDUCTED":
                account = "TDS Receivable" if s.amount > 0 else "TDS Payable"
                assert account in s.journal_entry
                assert s.journal_entry.startswith("Dr Bank ") == (s.amount < 0)   # TDS paid out comes back to Bank


def test_tds_scenario_is_opt_in():
    default, explicit = generator.generate(generator.GeneratorConfig()), generator.generate(
        generator.GeneratorConfig(n_tds=0))
    assert default.book == explicit.book and default.bank == explicit.bank
    assert "TDS_DEDUCTED" not in default.summary["items_by_category"]
    with_tds = generator.generate(generator.GeneratorConfig(n_tds=4))
    assert with_tds.summary["items_by_category"]["TDS_DEDUCTED"] == 4 and with_tds.summary["difference"] == "0.00"


def test_guardrail_only_allows_tds_when_the_arithmetic_fits(tds_months):
    run = tds_months[0]
    cases = {c.exception_id: c for c in run.cases}
    keying = next(cases[s.exception_id] for s in run.suggestions if s.category == "AMOUNT_ERROR")
    tds = next(cases[s.exception_id] for s in run.suggestions if s.category == "TDS_DEDUCTED")
    claim = Label(category="TDS_DEDUCTED", explanation="model says TDS", confidence=0.9)
    refused = resolve(keying, claim, "fake")
    assert refused.category == NEEDS_REVIEW and "fits a TDS rate" in refused.guardrail
    assert resolve(tds, claim, "fake").category == "TDS_DEDUCTED"


def test_evaluation_reports_the_tds_scenario(tmp_path):
    results = evaluate.evaluate(evaluate.months_ending("2026-06", 2), RulesLabeler(), tmp_path, tds=3)
    summary = results["summary"]
    assert summary["tds_per_month"] == 3 and summary["by_category"]["TDS_DEDUCTED"]["items"] == 6
    assert summary["match_precision_pct"] == 100.0


def test_llm_evidence_carries_the_tds_arithmetic_but_no_answers(tds_months):
    for case in tds_months[0].cases:
        if case.kind == PAIR:
            assert "tds_check" in case.evidence and "digits_swapped" in case.evidence
            assert "category" not in str(case.evidence)
