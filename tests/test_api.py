"""Segment 6 tests: the REST API, end to end through FastAPI's test client."""
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from recon import api, generator


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(api, "SAMPLE_DIR", tmp_path / "data")
    monkeypatch.setattr(api, "MEMORY_PATH", tmp_path / "payee_memory.json")
    api.RUNS.clear()
    # Seed 7: every exception has one right answer. (Seed 42's month has a keying error that also fits
    # TDS exactly, which goes to review; see test_tds.py.)
    generator.write_outputs(generator.generate(generator.GeneratorConfig(seed=7, tricky_narrations=False)),
                            tmp_path / "data")
    return TestClient(api.app)


def upload(client, folder, labeler="rules"):
    files = {name: (f"{name}.csv", (folder / f"{name}.csv").read_bytes(), "text/csv")
             for name in ("cash_book", "bank_statement", "parties")}
    return client.post("/reconciliations", files=files, data={"labeler": labeler})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_and_reconcile(client, tmp_path):
    response = upload(client, tmp_path / "data")
    assert response.status_code == 201
    body = response.json()
    assert body["statement"]["difference"] == "0.00" and body["matching"]["matches"] > 200
    assert body["scores"] is None                         # no answer key was uploaded
    assert len(client.get(f"/reconciliations/{body['run_id']}/matches").json()) == \
        body["matching"]["matches"] + body["matching"]["needs_correction"]


def test_sample_run_is_scored(client):
    body = client.post("/reconciliations/sample").json()
    assert body["scores"]["matching"]["match_precision_pct"] == 100.0


def test_review_workflow_and_statement(client):
    run_id = client.post("/reconciliations/sample").json()["run_id"]
    pending = client.get(f"/reconciliations/{run_id}/exceptions", params={"status": "PENDING_APPROVAL"}).json()
    assert pending and all(e["status"] == "PENDING_APPROVAL" for e in pending)
    charges = client.get(f"/reconciliations/{run_id}/exceptions", params={"category": "BANK_CHARGE"}).json()
    assert charges and all(e["category"] == "BANK_CHARGE" for e in charges)

    # nothing approved yet: the approved-only statement doesn't tie out
    assert client.get(f"/reconciliations/{run_id}/statement").json()["difference"] != "0.00"
    rejected = pending[0]
    assert client.post(f"/reconciliations/{run_id}/exceptions/{rejected['exception_id']}/reject",
                       json={"note": "checking with the bank"}).json()["status"] == "REJECTED"
    for item in pending[1:]:
        assert client.post(f"/reconciliations/{run_id}/exceptions/{item['exception_id']}/approve").status_code == 200
    statement = client.get(f"/reconciliations/{run_id}/statement").json()
    assert Decimal(statement["difference"]) == Decimal(rejected["adjustment"])   # only the rejected fix is missing
    again = client.post(f"/reconciliations/{run_id}/exceptions/{pending[1]['exception_id']}/approve")
    assert again.status_code == 409


def test_items_without_a_fix_cannot_be_approved(client):
    run_id = client.post("/reconciliations/sample").json()["run_id"]
    suggestion = api.RUNS[run_id]["run"].suggestions[0]
    suggestion.category = "NEEDS_REVIEW"
    response = client.post(f"/reconciliations/{run_id}/exceptions/{suggestion.exception_id}/approve")
    assert response.status_code == 409


def test_unknown_ids_and_bad_files(client, tmp_path):
    assert client.get("/reconciliations/nope").status_code == 404
    run_id = client.post("/reconciliations/sample").json()["run_id"]
    assert client.post(f"/reconciliations/{run_id}/exceptions/EX-999/approve").status_code == 404
    files = {"cash_book": ("b.csv", b"book_id,date\nBK-1,2026-06-01\n", "text/csv"),
             "bank_statement": ("k.csv", (tmp_path / "data" / "bank_statement.csv").read_bytes(), "text/csv")}
    response = client.post("/reconciliations", files=files)
    assert response.status_code == 422 and "missing columns" in response.json()["detail"]
    assert client.post("/reconciliations/sample", params={"labeler": "gpt"}).status_code == 422


def bank_only(client, run_id, category):
    return next(e for e in client.get(f"/reconciliations/{run_id}/exceptions", params={"category": category}).json()
                if e["kind"] == "BANK_ONLY")


def test_approving_a_bank_line_teaches_the_payee_memory(client):
    run_id = client.post("/reconciliations/sample").json()["run_id"]
    charge = bank_only(client, run_id, "BANK_CHARGE")
    assert client.post(f"/reconciliations/{run_id}/exceptions/{charge['exception_id']}/approve").status_code == 200
    learned = client.get("/payee-memory").json()
    assert len(learned) == 1 and learned[0]["category"] == "BANK_CHARGE" and learned[0]["approvals"] == 1

    second = client.post("/reconciliations/sample").json()
    assert second["labeler"] == "rules+memory"
    again = next(e for e in client.get(f"/reconciliations/{second['run_id']}/exceptions").json()
                 if e["description"] == charge["description"])
    assert again["category"] == "BANK_CHARGE" and "approved on" in again["explanation"]


def test_approving_with_a_corrected_account(client):
    run_id = client.post("/reconciliations/sample").json()["run_id"]
    debit = bank_only(client, run_id, "UNRECORDED_DEBIT")
    response = client.post(f"/reconciliations/{run_id}/exceptions/{debit['exception_id']}/approve",
                           json={"counter_account": "Professional Fees", "note": "it's the lawyer"})
    body = response.json()
    assert response.status_code == 200 and body["status"] == "APPROVED"
    assert body["journal_entry"].startswith("Dr Professional Fees ") and body["adjustment"] == debit["adjustment"]
    assert "Corrected by the reviewer" in body["explanation"] and body["reviewer_note"] == "it's the lawyer"
    assert [m["counter_account"] for m in client.get("/payee-memory").json()] == ["Professional Fees"]


def test_corrections_must_pass_the_guardrails(client):
    run_id = client.post("/reconciliations/sample").json()["run_id"]
    debit = bank_only(client, run_id, "UNRECORDED_DEBIT")
    url = f"/reconciliations/{run_id}/exceptions/{debit['exception_id']}/approve"
    assert client.post(url, json={"counter_account": "Party Expense"}).status_code == 422
    assert client.post(url, json={"category": "NOT_A_CATEGORY"}).status_code == 422
    assert client.post(url, json={"category": "INTEREST_CREDIT"}).status_code == 422     # money went out
    pair = client.get(f"/reconciliations/{run_id}/exceptions", params={"category": "AMOUNT_ERROR"}).json()[0]
    refused = client.post(f"/reconciliations/{run_id}/exceptions/{pair['exception_id']}/approve",
                          json={"category": "TDS_DEDUCTED"})
    assert refused.status_code == 422 and "TDS" in refused.json()["detail"]
    assert client.get("/payee-memory").json() == []
    assert client.get(f"/reconciliations/{run_id}/exceptions",
                      params={"status": "PENDING_APPROVAL"}).json()   # nothing was approved by the failures
