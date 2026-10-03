"""Segment 5 tests: the one-command run, its statement and its Power BI files."""
import csv
import json
from decimal import Decimal

import pytest

from recon import generator, pipeline
from recon.agent import RulesLabeler


def make_data(folder, seed=7, tricky=False):
    """Seed 7: every exception has one right answer (seed 42 has one that fits both a typo and TDS)."""
    generator.write_outputs(generator.generate(generator.GeneratorConfig(seed=seed, tricky_narrations=tricky)), folder)
    return folder


def run_on(folder, **kwargs):
    return pipeline.reconcile(folder / "cash_book.csv", folder / "bank_statement.csv", folder / "parties.csv",
                              RulesLabeler(), folder / "answer_key.csv", **kwargs)


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_correct_suggestions_make_the_statement_tie_out(tmp_path):
    data = make_data(tmp_path)
    statement = run_on(data).statement()
    expected = json.loads((data / "summary.json").read_text())      # the generator's own statement
    assert statement["difference"] == Decimal("0.00") and statement["items_needing_review"] == 0
    assert statement["adjusted_book_balance"] == Decimal(expected["adjusted_book_balance"])
    assert statement["adjusted_bank_balance"] == Decimal(expected["adjusted_bank_balance"])


def test_a_rejected_suggestion_leaves_exactly_its_amount_unreconciled(tmp_path):
    run = run_on(make_data(tmp_path))
    charge = next(s for s in run.suggestions if s.category == "BANK_CHARGE")
    charge.status = "REJECTED"
    assert run.statement()["difference"] == charge.adjustment


def test_review_items_are_counted_with_their_value(tmp_path):
    run = run_on(make_data(tmp_path))
    first = run.suggestions[0]
    first.category, first.adjusts, first.adjustment = "NEEDS_REVIEW", "NONE", Decimal("0.00")
    statement = run.statement()
    assert statement["items_needing_review"] == 1 and statement["value_needing_review"] == abs(first.amount)


def test_approved_only_statement_counts_just_the_approved(tmp_path):
    run = run_on(make_data(tmp_path))
    assert run.statement(approved_only=True)["suggestions_counted"] == 0
    run.suggestions[0].status = "APPROVED"
    assert run.statement(approved_only=True)["suggestions_counted"] == 1


def test_writes_power_bi_files_with_one_row_per_line(tmp_path):
    data = make_data(tmp_path / "data")
    run = run_on(data)
    pipeline.write_outputs(run, tmp_path / "out")
    lines = rows(tmp_path / "out" / "lines.csv")
    assert len(lines) == len(run.book_lines) + len(run.bank_lines)
    assert {r["status"] for r in lines} == {"MATCHED", "NEEDS_CORRECTION", "EXCEPTION"}
    assert all(r["age_bucket"] for r in lines if r["status"] != "MATCHED")
    assert len(rows(tmp_path / "out" / "exceptions.csv")) == len(run.suggestions)
    assert len(rows(tmp_path / "out" / "matches.csv")) == len(run.result.matches)
    statement = rows(tmp_path / "out" / "reconciliation.csv")
    assert statement[-1]["line"] == "Difference" and statement[-1]["amount"] == "0.00"
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["scores"]["matching"]["match_precision_pct"] == 100.0


def test_runs_without_an_answer_key(tmp_path):
    data = make_data(tmp_path)
    (data / "answer_key.csv").unlink()
    run = pipeline.reconcile(data / "cash_book.csv", data / "bank_statement.csv", data / "parties.csv")
    assert run.scores is None and run.statement()["difference"] == Decimal("0.00")


def test_output_is_identical_on_every_run(tmp_path):
    data = make_data(tmp_path / "data")
    for folder in ("a", "b"):
        pipeline.write_outputs(run_on(data), tmp_path / folder)
    for name in ("lines.csv", "matches.csv", "exceptions.csv", "reconciliation.csv", "summary.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_command_line(tmp_path, capsys):
    data = make_data(tmp_path / "data")
    pipeline.main(["--data", str(data), "--out", str(tmp_path / "out"), "--labeler", "rules"])
    printed = capsys.readouterr().out
    assert "match precision 100.0%" in printed and (tmp_path / "out" / "exceptions.csv").exists()


def test_report_warns_when_the_labeler_fails(tmp_path):
    class BrokenLabeler:
        name = "broken"

        def label_all(self, cases):
            return [ConnectionError("no network") for _ in cases]

    data = make_data(tmp_path)
    run = pipeline.reconcile(data / "cash_book.csv", data / "bank_statement.csv", data / "parties.csv",
                             BrokenLabeler(), data / "answer_key.csv")
    text = pipeline.report(run)
    assert "WARNING" in text and "ConnectionError" in text
    assert all(s.category == "NEEDS_REVIEW" for s in run.suggestions)


def test_make_labeler_builds_the_hybrid_without_calling_a_model():
    pytest.importorskip("langchain_ollama")
    assert {"hybrid-ollama", "hybrid-openai"} <= set(pipeline.LABELERS)
    assert pipeline.make_labeler("hybrid-ollama", model="llama3.1:8b").name == "hybrid:ollama:llama3.1:8b"
