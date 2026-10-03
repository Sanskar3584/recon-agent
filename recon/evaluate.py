"""Evaluate the matcher and a labeler over many generated months.

    python -m recon.evaluate --months 12 --labeler rules
    python -m recon.evaluate --months 12 --labeler openai
    python -m recon.evaluate --months 12 --labeler hybrid-ollama
    python -m recon.evaluate --months 12 --labeler rules --tds 3      # add 3 TDS settlements a month
    python -m recon.evaluate --months 12 --labeler rules --memory     # carry approvals month to month

Each month is generated fresh (its own seed and calendar month), reconciled and scored
against its answer key. Per-month results go to output/evaluation.csv, the totals and the
per-category accuracy to output/evaluation_summary.json.

With --memory, a simulated reviewer approves the right label for every bank-only line at the end
of each month (correcting the suggestion where it was wrong, as a person would), and those
approvals go into a payee memory that labels the following months. Months run in calendar order.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

from recon import generator, pipeline
from recon.memory import MemoryLabeler, PayeeMemory
from recon.score import load_answer_key, reviewer_approvals


def months_ending(end: str, count: int) -> list:
    year, month = (int(part) for part in end.split("-"))
    months = []
    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year, month - 1) if month > 1 else (year - 1, 12)
    return list(reversed(months))


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 1) if whole else 100.0


def evaluate(months: list, labeler, workdir, tds: int = 0, memory: PayeeMemory = None) -> dict:
    """Score `labeler` over `months`. `tds` adds that many TDS settlements to each month; `memory`
    (a PayeeMemory) turns on the payee memory with a simulated reviewer feeding it."""
    workdir = Path(workdir)
    per_month, categories = [], {}
    if memory is not None:
        labeler = MemoryLabeler(labeler, memory)
    totals = {"lines": 0, "proposed_pairs": 0, "correct_pairs": 0, "matchable": 0, "matched_right": 0,
              "exceptions": 0, "category_right": 0, "entry_right": 0, "needs_review": 0}
    for index, month in enumerate(months):
        folder = workdir / month
        generator.write_outputs(generator.generate(generator.GeneratorConfig(month=month, seed=1000 + index, n_tds=tds)),
                                folder)
        run = pipeline.reconcile(folder / "cash_book.csv", folder / "bank_statement.csv", folder / "parties.csv",
                                 labeler, folder / "answer_key.csv")
        ms, ls = run.scores["matching"], run.scores["labels"]
        if memory is not None:
            banks = {line.bank_id: line for line in run.bank_lines}
            for bank_id, category, account in reviewer_approvals(run.suggestions,
                                                                 load_answer_key(folder / "answer_key.csv")):
                memory.remember(banks[bank_id], category, account, run.result.period_end.isoformat())
        matched_right = sum(ms["by_category"].get(c, {}).get("correct", 0) for c in pipeline_match_categories())
        category_right = sum(v["correct"] for v in ls["by_category"].values())
        entry_right = sum(v["entry_correct"] for v in ls["by_category"].values())
        for category, stats in ls["by_category"].items():
            bucket = categories.setdefault(category, {"items": 0, "correct": 0, "entry_correct": 0})
            for key in bucket:
                bucket[key] += stats[key]
        lines = len(run.book_lines) + len(run.bank_lines)
        totals["lines"] += lines
        totals["proposed_pairs"] += ms["proposed_pairs"]
        totals["correct_pairs"] += ms["correct_pairs"]
        totals["matchable"] += ms["matchable_items"]
        totals["matched_right"] += matched_right
        totals["exceptions"] += ls["exceptions"]
        totals["category_right"] += category_right
        totals["entry_right"] += entry_right
        totals["needs_review"] += ls["needs_review"]
        per_month.append({"labeler": labeler.name, "month": month, "lines": lines, "match_precision_pct": ms["match_precision_pct"],
                          "auto_match_rate_pct": ms["auto_match_rate_pct"], "exceptions": ls["exceptions"],
                          "category_accuracy_pct": ls["category_accuracy_pct"],
                          "entry_accuracy_pct": ls["entry_accuracy_pct"], "needs_review": ls["needs_review"],
                          "difference": f"{run.statement()['difference']:.2f}"})
    summary = {
        "labeler": labeler.name, "months": len(months), "first_month": months[0], "last_month": months[-1],
        "lines": totals["lines"],
        "match_precision_pct": _pct(totals["correct_pairs"], totals["proposed_pairs"]),
        "auto_match_rate_pct": _pct(totals["matched_right"], totals["matchable"]),
        "exceptions": totals["exceptions"],
        "category_accuracy_pct": _pct(totals["category_right"], totals["exceptions"]),
        "entry_accuracy_pct": _pct(totals["entry_right"], totals["exceptions"]),
        "needs_review": totals["needs_review"],
        "by_category": {c: dict(v, accuracy_pct=_pct(v["entry_correct"], v["items"]))
                        for c, v in sorted(categories.items())},
    }
    if tds:
        summary["tds_per_month"] = tds
    if memory is not None:
        summary["answered_from_memory"] = labeler.hits
    return {"summary": summary, "per_month": per_month}


def pipeline_match_categories() -> list:
    from recon.score import MATCH_CATEGORIES
    return MATCH_CATEGORIES


def write_evaluation(results: dict, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = results["per_month"]
    with (out / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "evaluation_summary.json").open("w", newline="\n", encoding="utf-8") as handle:
        json.dump(results["summary"], handle, indent=2)
        handle.write("\n")


def report(summary: dict) -> str:
    lines = [f"{summary['labeler']} over {summary['months']} months ({summary['first_month']} to {summary['last_month']}), "
             f"{summary['lines']:,} book and bank lines",
             f"  matcher: precision {summary['match_precision_pct']}%, auto-match rate {summary['auto_match_rate_pct']}%",
             f"  exceptions: {summary['exceptions']}; category right {summary['category_accuracy_pct']}%, "
             f"journal entry exactly right {summary['entry_accuracy_pct']}%, sent to review {summary['needs_review']}"
             + (f", answered from payee memory {summary['answered_from_memory']}" if "answered_from_memory" in summary else ""),
             "", f"  {'category':<22}{'items':>6}{'right':>7}{'accuracy':>10}"]
    for category, stats in summary["by_category"].items():
        lines.append(f"  {category:<22}{stats['items']:>6}{stats['entry_correct']:>7}{stats['accuracy_pct']:>9.1f}%")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Score the matcher and a labeler over many generated months.")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--end", default="2026-06", help="last month, YYYY-MM")
    parser.add_argument("--labeler", default="rules", choices=pipeline.LABELERS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default="output")
    parser.add_argument("--tds", type=int, default=0, help="TDS settlements to add to each month (default 0)")
    parser.add_argument("--memory", action="store_true",
                        help="carry a simulated reviewer's approvals from month to month in a payee memory")
    args = parser.parse_args(argv)
    labeler = pipeline.make_labeler(args.labeler, args.model)
    results = evaluate(months_ending(args.end, args.months), labeler, Path(args.out) / "evaluation_data",
                       tds=args.tds, memory=PayeeMemory() if args.memory else None)
    write_evaluation(results, args.out)
    print(report(results["summary"]))
    print(f"Per-month results in {args.out}/evaluation.csv")


if __name__ == "__main__":
    main()
