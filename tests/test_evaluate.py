"""Segment 5 tests: the multi-month evaluation."""
import json

from recon import evaluate
from recon.agent import RulesLabeler


def test_months_ending_counts_back_across_year_ends():
    assert evaluate.months_ending("2026-02", 4) == ["2025-11", "2025-12", "2026-01", "2026-02"]


def test_evaluates_several_months_and_aggregates(tmp_path):
    results = evaluate.evaluate(evaluate.months_ending("2026-06", 3), RulesLabeler(), tmp_path / "data")
    summary = results["summary"]
    assert summary["months"] == 3 and len(results["per_month"]) == 3
    assert summary["match_precision_pct"] == 100.0 and summary["auto_match_rate_pct"] == 100.0
    assert summary["exceptions"] == sum(m["exceptions"] for m in results["per_month"])
    assert sum(v["items"] for v in summary["by_category"].values()) == summary["exceptions"]
    evaluate.write_evaluation(results, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "evaluation_summary.json").read_text())["months"] == 3
