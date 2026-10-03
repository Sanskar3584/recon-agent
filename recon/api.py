"""Segment 6: a small REST API around the pipeline.

    uvicorn recon.api:app --reload        then open http://127.0.0.1:8000/docs

Runs are kept in memory and their files are written under runs/<run_id>/. Approved labels for
bank-only lines go into payee_memory.json, and later runs label the same payees from it. This is
a local demo: there is no login, so don't expose it to a network as it is.
"""
from __future__ import annotations

import shutil
import uuid
from dataclasses import asdict, is_dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from recon import pipeline
from recon.agent import BANK_ONLY, NEEDS_REVIEW, PENDING, Label, resolve
from recon.loader import LoadError
from recon.memory import PayeeMemory

PROJECT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT / "runs"
SAMPLE_DIR = PROJECT / "data"
MEMORY_PATH = PROJECT / "payee_memory.json"
RUNS: dict = {}

app = FastAPI(title="Bank Reconciliation Agent",
              description="Match a cash book to a bank statement, label the exceptions, approve the fixes.")


def _plain(value):
    """JSON-safe copy: Decimals as exact strings, dates as ISO strings."""
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    if isinstance(value, date):
        return value.isoformat()
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, f"No reconciliation {run_id}")
    return RUNS[run_id]


def _overview(run_id: str) -> dict:
    run = _run(run_id)["run"]
    return _plain({"run_id": run_id, "labeler": run.labeler, "matching": run.result.summary(),
                   "statement": run.statement(), "scores": run.scores})


def _start(folder: Path, labeler_name: str) -> dict:
    run_id = folder.name
    try:
        run = pipeline.reconcile(folder / "cash_book.csv", folder / "bank_statement.csv",
                                 folder / "parties.csv",
                                 pipeline.make_labeler(labeler_name, memory=PayeeMemory(MEMORY_PATH)),
                                 folder / "answer_key.csv")
    except LoadError as error:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(422, str(error)) from None
    pipeline.write_outputs(run, folder / "output")
    RUNS[run_id] = {"run": run, "folder": folder}
    return _overview(run_id)


def _new_folder() -> Path:
    folder = RUNS_DIR / uuid.uuid4().hex[:8]
    folder.mkdir(parents=True)
    return folder


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reconciliations", status_code=201)
async def create_reconciliation(cash_book: UploadFile = File(...), bank_statement: UploadFile = File(...),
                                parties: Optional[UploadFile] = File(None), labeler: str = Form("rules")):
    """Upload a cash book and a bank statement (CSV, same columns as the generator writes)."""
    if labeler not in pipeline.LABELERS:
        raise HTTPException(422, f"labeler must be one of: {', '.join(pipeline.LABELERS)}")
    folder = _new_folder()
    for upload, name in ((cash_book, "cash_book.csv"), (bank_statement, "bank_statement.csv"), (parties, "parties.csv")):
        if upload is not None:
            (folder / name).write_bytes(await upload.read())
    return _start(folder, labeler)


@app.post("/reconciliations/sample", status_code=201)
def create_sample_reconciliation(labeler: str = "rules"):
    """Reconcile the sample month in data/ (and score it against its answer key)."""
    if labeler not in pipeline.LABELERS:
        raise HTTPException(422, f"labeler must be one of: {', '.join(pipeline.LABELERS)}")
    folder = _new_folder()
    for name in ("cash_book.csv", "bank_statement.csv", "parties.csv", "answer_key.csv"):
        shutil.copy(SAMPLE_DIR / name, folder / name)
    return _start(folder, labeler)


@app.get("/reconciliations/{run_id}")
def get_reconciliation(run_id: str):
    return _overview(run_id)


@app.get("/reconciliations/{run_id}/matches")
def list_matches(run_id: str, rule: Optional[str] = None):
    matches = _run(run_id)["run"].result.matches
    return _plain([dict(asdict(m), difference=m.difference) for m in matches if rule in (None, m.rule)])


@app.get("/reconciliations/{run_id}/exceptions")
def list_exceptions(run_id: str, status: Optional[str] = None, category: Optional[str] = None):
    suggestions = _run(run_id)["run"].suggestions
    return _plain([s for s in suggestions if status in (None, s.status) and category in (None, s.category)])


def _pending(run_id: str, exception_id: str):
    run = _run(run_id)["run"]
    suggestion = next((s for s in run.suggestions if s.exception_id == exception_id), None)
    if suggestion is None:
        raise HTTPException(404, f"No exception {exception_id} in {run_id}")
    if suggestion.status != PENDING:
        raise HTTPException(409, f"{exception_id} is already {suggestion.status}")
    return run, suggestion


def _correct(run_id: str, exception_id: str, category: Optional[str], counter_account: Optional[str]) -> None:
    """Replace the suggestion with the reviewer's category and/or account. Python recomputes the
    amounts and the journal entry, and the same guardrails apply as to any labeler."""
    run, current = _pending(run_id, exception_id)
    case = next(c for c in run.cases if c.exception_id == exception_id)
    try:
        was = " ".join(part for part in (current.category, current.counter_account) if part)
        label = Label(category=category or current.category,
                      counter_account=current.counter_account if counter_account is None else counter_account,
                      explanation=f"Corrected by the reviewer (suggested: {was}).", confidence=1.0)
    except ValidationError:
        raise HTTPException(422, f"{category!r} is not a category") from None
    fixed = resolve(case, label, "reviewer")
    if fixed.category == NEEDS_REVIEW:
        raise HTTPException(422, f"That correction fails a guardrail: {fixed.guardrail or 'no fix to approve'}")
    run.suggestions[run.suggestions.index(current)] = fixed


def _remember(run_id: str, exception_id: str) -> None:
    """Put an approved bank-only label into the payee memory."""
    run = _run(run_id)["run"]
    s = next(s for s in run.suggestions if s.exception_id == exception_id)
    if s.kind != BANK_ONLY:
        return
    line = next(l for l in run.bank_lines if l.bank_id == s.bank_ids[0])
    account = s.counter_account if s.category in ("UNRECORDED_DEBIT", "DIRECT_CREDIT") else ""
    memory = PayeeMemory(MEMORY_PATH)
    if memory.remember(line, s.category, account, date.today().isoformat()):
        memory.save()


def _decide(run_id: str, exception_id: str, decision: str, note: str) -> dict:
    run, suggestion = _pending(run_id, exception_id)
    if decision == "APPROVED" and suggestion.category == NEEDS_REVIEW:
        raise HTTPException(409, f"{exception_id} has no suggested fix; investigate it and reject or re-run")
    suggestion.status, suggestion.reviewer_note = decision, note
    pipeline.write_outputs(run, _run(run_id)["folder"] / "output")   # keep the Power BI files current
    return _plain(suggestion)


@app.post("/reconciliations/{run_id}/exceptions/{exception_id}/approve")
def approve(run_id: str, exception_id: str, note: str = Body("", embed=True),
            category: Optional[str] = Body(None, embed=True), counter_account: Optional[str] = Body(None, embed=True)):
    """Approve the suggested fix, optionally correcting its category or account first. Approved
    bank-only lines are remembered, so next time the same payee is labeled the way you approved it."""
    if category is not None or counter_account is not None:
        _correct(run_id, exception_id, category, counter_account)
    decided = _decide(run_id, exception_id, "APPROVED", note)
    _remember(run_id, exception_id)
    return decided


@app.post("/reconciliations/{run_id}/exceptions/{exception_id}/reject")
def reject(run_id: str, exception_id: str, note: str = Body("", embed=True)):
    return _decide(run_id, exception_id, "REJECTED", note)


@app.get("/payee-memory")
def payee_memory():
    """Every payee the memory has learned from approvals, with the label it will apply."""
    return [dict(key=key, **vars(entry)) for key, entry in sorted(PayeeMemory(MEMORY_PATH).entries.items())]


@app.get("/reconciliations/{run_id}/statement")
def statement(run_id: str, approved_only: bool = True):
    """The bank reconciliation statement. By default only approved fixes count."""
    return _plain(_run(run_id)["run"].statement(approved_only=approved_only))
