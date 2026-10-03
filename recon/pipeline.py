"""Segment 5: run the whole reconciliation with one command and write Power BI-ready files.

    python -m recon.pipeline --labeler rules      # keyword rules, no API
    python -m recon.pipeline --labeler openai     # needs OPENAI_API_KEY in .env
    python -m recon.pipeline --labeler ollama     # needs Ollama running locally
    python -m recon.pipeline --labeler hybrid-ollama   # rules, plus Ollama for bank-only lines
    python -m recon.pipeline --memory payee_memory.json  # answer recurring payees from past approvals

It reads cash_book.csv, bank_statement.csv and parties.csv from --data, matches the lines,
labels the exceptions, builds the bank reconciliation statement and writes CSVs to --out.
If answer_key.csv is in --data, the run is also scored. The key is read only for scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional

from recon.agent import NEEDS_REVIEW, HybridLabeler, LLMLabeler, RulesLabeler, build_exceptions, label_exceptions
from recon.loader import load_bank, load_book, load_parties, opening_balance
from recon.matcher import MATCHED, match
from recon.memory import MemoryLabeler, PayeeMemory
from recon.score import load_answer_key, score_labels, score_matching

ZERO = Decimal("0.00")


@dataclass
class Run:
    book_lines: list
    bank_lines: list
    result: object              # MatchResult
    suggestions: list
    book_opening: Decimal
    bank_opening: Decimal
    labeler: str
    scores: Optional[dict] = None
    cases: Optional[list] = None   # the exception cases with their evidence, for corrections made later

    def statement(self, approved_only: bool = False) -> dict:
        """The bank reconciliation statement. By default every pending suggestion counts;
        with approved_only=True only the ones a reviewer approved do."""
        use = [s for s in self.suggestions if s.status == "APPROVED" or (not approved_only and s.status == "PENDING_APPROVAL")]
        book_closing = self.book_opening + sum((l.amount for l in self.book_lines), ZERO)
        bank_closing = self.bank_opening + sum((l.amount for l in self.bank_lines), ZERO)
        book_adjustments = sum((s.adjustment for s in use if s.adjusts == "BOOK"), ZERO)
        bank_adjustments = sum((s.adjustment for s in use if s.adjusts == "BANK"), ZERO)
        in_transit = sum((s.adjustment for s in use if s.category == "DEPOSIT_IN_TRANSIT"), ZERO)
        outstanding = sum((s.adjustment for s in use if s.category == "OUTSTANDING_PAYMENT"), ZERO)
        review = [s for s in self.suggestions if s.category == NEEDS_REVIEW]
        adjusted_book, adjusted_bank = book_closing + book_adjustments, bank_closing + bank_adjustments
        return {
            "period_end": self.result.period_end.isoformat(),
            "book_closing_balance": book_closing, "book_adjustments": book_adjustments,
            "adjusted_book_balance": adjusted_book,
            "bank_closing_balance": bank_closing, "deposits_in_transit": in_transit,
            "outstanding_payments": outstanding, "adjusted_bank_balance": adjusted_bank,
            "difference": adjusted_bank - adjusted_book,
            "items_needing_review": len(review),
            "value_needing_review": sum((abs(s.amount) for s in review), ZERO),
            "suggestions_counted": len(use), "approved_only": approved_only,
        }


LABELERS = ["rules", "openai", "ollama", "hybrid-openai", "hybrid-ollama"]


def make_labeler(name: str, model: Optional[str] = None, memory: Optional[PayeeMemory] = None):
    """Build a labeler by name. With a payee memory, recurring payees are answered from it first."""
    if name == "rules":
        labeler = RulesLabeler()
    elif name.startswith("hybrid-"):
        labeler = HybridLabeler(LLMLabeler(name[len("hybrid-"):], model=model))
    else:
        labeler = LLMLabeler(name, model=model)
    return MemoryLabeler(labeler, memory) if memory is not None else labeler


def reconcile(book_path, bank_path, parties_path=None, labeler=None, answer_key_path=None) -> Run:
    book_lines, bank_lines = load_book(book_path), load_bank(bank_path)
    if parties_path and Path(parties_path).exists():
        customers = [p for p, kind in load_parties(parties_path).items() if kind == "CUSTOMER"]
    else:
        customers = sorted({l.party for l in book_lines if l.amount > 0})
    result = match(book_lines, bank_lines)
    labeler = labeler or RulesLabeler()
    cases = build_exceptions(result, book_lines, bank_lines, customers)
    suggestions = label_exceptions(cases, labeler)
    run = Run(book_lines, bank_lines, result, suggestions,
              opening_balance(book_path, "receipt", "payment"),
              opening_balance(bank_path, "deposit", "withdrawal"), labeler.name, cases=cases)
    if answer_key_path and Path(answer_key_path).exists():
        key = load_answer_key(answer_key_path)
        run.scores = {"matching": score_matching(result, key), "labels": score_labels(suggestions, key)}
    return run


# ---------------------------------------------------------------------------------------
# Output files, shaped for Power BI (one fact table per grain, plain columns)
# ---------------------------------------------------------------------------------------
def age_bucket(days: int) -> str:
    if days <= 7:
        return "0-7 days"
    if days <= 15:
        return "8-15 days"
    if days <= 30:
        return "16-30 days"
    return "Over 30 days"


def _fmt(value):
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    if isinstance(value, list):
        return ";".join(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return "" if value is None else value


def _write(path: Path, rows: list, columns: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_fmt(row[c]) for c in columns])


def line_rows(run: Run) -> list:
    status, match_of, rule_of, exception_of, category_of = {}, {}, {}, {}, {}
    for m in run.result.matches:
        for line_id in m.book_ids + m.bank_ids:
            status[line_id] = "MATCHED" if m.status == MATCHED else "NEEDS_CORRECTION"
            match_of[line_id], rule_of[line_id] = m.match_id, m.rule
    for s in run.suggestions:
        for line_id in s.book_ids + s.bank_ids:
            exception_of[line_id], category_of[line_id] = s.exception_id, s.category
            status.setdefault(line_id, "EXCEPTION")
    period_end = run.result.period_end
    rows = []
    for side, lines in (("BOOK", run.book_lines), ("BANK", run.bank_lines)):
        for l in lines:
            line_id = l.book_id if side == "BOOK" else l.bank_id
            day = l.date if side == "BOOK" else l.value_date
            is_open = status.get(line_id) != "MATCHED"
            rows.append({
                "line_id": line_id, "side": side, "date": day,
                "description": l.particulars if side == "BOOK" else l.narration,
                "reference": l.reference if side == "BOOK" else l.bank_ref,
                "direction": "IN" if l.amount > 0 else "OUT", "amount": l.amount, "abs_amount": abs(l.amount),
                "status": status.get(line_id, "EXCEPTION"), "match_id": match_of.get(line_id, ""),
                "match_rule": rule_of.get(line_id, ""), "exception_id": exception_of.get(line_id, ""),
                "category": category_of.get(line_id, "MATCHED" if not is_open else ""),
                "age_days": (period_end - day).days if is_open else "",
                "age_bucket": age_bucket((period_end - day).days) if is_open else "",
            })
    return rows


LINE_COLUMNS = ["line_id", "side", "date", "description", "reference", "direction", "amount", "abs_amount",
                "status", "match_id", "match_rule", "exception_id", "category", "age_days", "age_bucket"]
MATCH_COLUMNS = ["match_id", "rule", "status", "book_ids", "bank_ids", "book_amount", "bank_amount",
                 "difference", "score", "detail"]
EXCEPTION_COLUMNS = ["exception_id", "kind", "category", "book_ids", "bank_ids", "line_date", "description",
                     "amount", "abs_amount", "adjusts", "adjustment", "counter_account", "journal_entry",
                     "explanation", "confidence", "labeler", "guardrail", "status", "age_days", "age_bucket"]


def write_outputs(run: Run, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    period_end = run.result.period_end
    _write(out / "lines.csv", line_rows(run), LINE_COLUMNS)
    _write(out / "matches.csv", [dict(vars(m), difference=m.difference) for m in run.result.matches], MATCH_COLUMNS)
    _write(out / "exceptions.csv",
           [dict(vars(s), abs_amount=abs(s.amount), age_days=(period_end - s.line_date).days,
                 age_bucket=age_bucket((period_end - s.line_date).days)) for s in run.suggestions],
           EXCEPTION_COLUMNS)
    st = run.statement()
    _write(out / "reconciliation.csv", [
        {"order": 1, "section": "Book", "line": "Balance as per cash book", "amount": st["book_closing_balance"]},
        {"order": 2, "section": "Book", "line": "Add/(less) suggested book adjustments", "amount": st["book_adjustments"]},
        {"order": 3, "section": "Book", "line": "Adjusted book balance", "amount": st["adjusted_book_balance"]},
        {"order": 4, "section": "Bank", "line": "Balance as per bank statement", "amount": st["bank_closing_balance"]},
        {"order": 5, "section": "Bank", "line": "Add deposits in transit", "amount": st["deposits_in_transit"]},
        {"order": 6, "section": "Bank", "line": "Less outstanding payments", "amount": st["outstanding_payments"]},
        {"order": 7, "section": "Bank", "line": "Adjusted bank balance", "amount": st["adjusted_bank_balance"]},
        {"order": 8, "section": "Check", "line": "Difference", "amount": st["difference"]},
    ], ["order", "section", "line", "amount"])
    summary = {"labeler": run.labeler, "matching": run.result.summary(),
               "statement": {k: _fmt(v) for k, v in st.items()},
               "exceptions_by_category": _count(s.category for s in run.suggestions)}
    if run.scores:
        summary["scores"] = run.scores
    with (out / "summary.json").open("w", newline="\n", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
        handle.write("\n")


def _count(values) -> dict:
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items()))


def report(run: Run) -> str:
    m, st = run.result.summary(), run.statement()
    lines = [f"Labeler: {run.labeler}",
             f"Matched {m['matches']} pairs; {m['needs_correction']} need correction; "
             f"{m['unmatched_book_lines']} book and {m['unmatched_bank_lines']} bank lines left as exceptions",
             f"Lines auto-matched: {m['lines_auto_matched_pct']}%",
             f"Adjusted book {st['adjusted_book_balance']:,.2f} vs adjusted bank {st['adjusted_bank_balance']:,.2f}"
             f"  difference {st['difference']:,.2f}  ({st['items_needing_review']} item(s) need review)"]
    failed = [s for s in run.suggestions if s.guardrail.startswith("labeler failed")]
    if failed:
        lines.append(f"WARNING: {len(failed)} of {len(run.suggestions)} labels failed and went to review. "
                     f"First error: {failed[0].guardrail}")
    if run.scores:
        ms, ls = run.scores["matching"], run.scores["labels"]
        lines += [f"Score vs answer key: match precision {ms['match_precision_pct']}%, "
                  f"auto-match rate {ms['auto_match_rate_pct']}%, exceptions surfaced {ms['exceptions_surfaced_pct']}%",
                  f"                     exception category accuracy {ls['category_accuracy_pct']}%, "
                  f"journal entries exactly right {ls['entry_accuracy_pct']}% ({ls['exceptions']} exceptions)"]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Reconcile a cash book against a bank statement.")
    parser.add_argument("--data", default="data", help="folder with cash_book.csv and bank_statement.csv")
    parser.add_argument("--out", default="output", help="folder for the results")
    parser.add_argument("--labeler", default="rules", choices=LABELERS)
    parser.add_argument("--model", default=None, help="override the LLM model name")
    parser.add_argument("--memory", default=None, help="payee memory file to label recurring payees from")
    args = parser.parse_args(argv)
    data = Path(args.data)
    labeler = make_labeler(args.labeler, args.model, PayeeMemory(args.memory) if args.memory else None)
    run = reconcile(data / "cash_book.csv", data / "bank_statement.csv", data / "parties.csv",
                    labeler, data / "answer_key.csv")
    write_outputs(run, args.out)
    print(report(run))
    if args.memory:
        print(f"Answered from payee memory: {labeler.hits} line(s)")
    print(f"Results written to {args.out}/")


if __name__ == "__main__":
    main()
