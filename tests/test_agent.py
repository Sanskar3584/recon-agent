"""Segment 4 tests: evidence, the rules labeler, guardrails and the LLM wiring (no network)."""
import json

import pytest

from recon import generator
from recon.agent import (BANK_ONLY, BOOK_ONLY, NEEDS_REVIEW, PAIR, PENDING, HybridLabeler, Label, LLMLabeler,
                         RulesLabeler, build_exceptions, label_exceptions, resolve)
from recon.loader import load_bank, load_book, load_parties
from recon.matcher import match
from recon.score import BANK_ONLY_CATEGORIES, BOOK_ONLY_CATEGORIES, load_answer_key, score_labels


def prepare(folder, seed=42, tricky=True):
    config = generator.GeneratorConfig(seed=seed, tricky_narrations=tricky)
    generator.write_outputs(generator.generate(config), folder)
    books, banks = load_book(folder / "cash_book.csv"), load_bank(folder / "bank_statement.csv")
    customers = [p for p, kind in load_parties(folder / "parties.csv").items() if kind == "CUSTOMER"]
    return build_exceptions(match(books, banks), books, banks, customers), load_answer_key(folder / "answer_key.csv")


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    return prepare(tmp_path_factory.mktemp("data"))


def label(category, account="", text="stub"):
    return Label(category=category, counter_account=account, explanation=text, confidence=0.9)


def find(cases, kind, money_in=None, word=""):
    return next(c for c in cases[0] if c.kind == kind and (money_in is None or c.money_in == money_in)
                and word in json.dumps(c.evidence))


def test_one_case_per_exception_with_no_answer_key_data(cases):
    exceptions, key = cases
    kinds = [c.kind for c in exceptions]
    assert kinds.count(PAIR) == sum(i["category"] == "AMOUNT_ERROR" for i in key)
    assert kinds.count(BOOK_ONLY) == sum(i["category"] in BOOK_ONLY_CATEGORIES for i in key)
    assert kinds.count(BANK_ONLY) == sum(i["category"] in BANK_ONLY_CATEGORIES for i in key)
    for case in exceptions:
        text = json.dumps(case.evidence)
        assert "AK-" not in text and "suggested_entry" not in text and "category" not in text


@pytest.mark.parametrize("seed", [42, 1, 7, 99, 2024])
def test_rules_labeler_is_right_when_narrations_are_plain(tmp_path, seed):
    """Every label is right, except a keying error whose shortfall also fits TDS exactly (seed 42 has
    one): with two explanations that fit the numbers, the rules send it to review instead of guessing."""
    exceptions, key = prepare(tmp_path, seed, tricky=False)
    suggestions = label_exceptions(exceptions, RulesLabeler())
    ambiguous = {s.exception_id for s in suggestions if s.category == NEEDS_REVIEW}
    for case in exceptions:
        if case.exception_id in ambiguous:
            assert case.kind == PAIR and "fits TDS" in case.evidence["tds_check"] and case.evidence["digits_swapped"]
    scores = score_labels([s for s in suggestions if s.exception_id not in ambiguous], key)
    assert scores["category_accuracy_pct"] == 100.0 and scores["entry_accuracy_pct"] == 100.0
    assert len(ambiguous) == (1 if seed == 42 else 0)


def test_rules_labeler_struggles_with_tricky_narrations(tmp_path):
    """The reason for an LLM: keyword rules misread lines such as LOCKER RENT or a shortened remitter name."""
    results = []
    for seed in range(20):
        exceptions, key = prepare(tmp_path / str(seed), seed, tricky=True)
        results.append(score_labels(label_exceptions(exceptions, RulesLabeler()), key)["entry_accuracy_pct"])
    assert min(results) < 100.0


def test_impossible_categories_go_to_review(cases):
    s = resolve(find(cases, BANK_ONLY), label("DEPOSIT_IN_TRANSIT"), "fake")
    assert s.category == NEEDS_REVIEW and "not possible" in s.guardrail and s.adjustment == 0


def test_direction_is_enforced(cases):
    s = resolve(find(cases, BANK_ONLY, money_in=False), label("INTEREST_CREDIT"), "fake")
    assert s.category == NEEDS_REVIEW


def test_accounts_must_come_from_the_chart_or_the_customer_master(cases):
    debit = find(cases, BANK_ONLY, money_in=False, word="NACH")
    assert resolve(debit, label("UNRECORDED_DEBIT", "Party Expense"), "fake").category == NEEDS_REVIEW
    good = resolve(debit, label("UNRECORDED_DEBIT", "Insurance Expense"), "fake")
    assert good.category == "UNRECORDED_DEBIT" and good.journal_entry.startswith("Dr Insurance Expense ")
    credit = find(cases, BANK_ONLY, money_in=True, word="ADVANCE")
    assert resolve(credit, label("DIRECT_CREDIT", "Customer Advances - Somebody Else"), "fake").category == NEEDS_REVIEW


def test_amounts_come_from_the_lines_not_the_labeler(cases):
    charge = find(cases, BANK_ONLY, money_in=False, word="CHRGS")
    s = resolve(charge, label("BANK_CHARGE", text="This charge is Rs 1,00,000"), "fake")
    assert s.adjustment == charge.bank[0].amount and s.journal_entry.count(f"{abs(charge.bank[0].amount):.2f}") == 2


def test_labeler_failures_become_review_items(cases):
    s = resolve(cases[0][0], TimeoutError("API timed out"), "fake")
    assert s.category == NEEDS_REVIEW and "TimeoutError" in s.guardrail


def test_every_suggestion_waits_for_approval(cases):
    assert {s.status for s in label_exceptions(cases[0], RulesLabeler())} == {PENDING}


def test_llm_labeler_builds_without_calling_the_api(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    assert LLMLabeler("openai", model="gpt-6-luna").name == "openai:gpt-6-luna"
    with pytest.raises(ValueError):
        LLMLabeler("not-a-provider")


def test_llm_labeler_sends_the_evidence_and_applies_guardrails(cases, monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    labeler, sent = LLMLabeler("openai"), []

    class StubChain:
        def batch(self, prompts, config=None, return_exceptions=False):
            sent.extend(prompts)
            return [label("DEPOSIT_IN_TRANSIT") for _ in prompts]   # wrong for most cases on purpose

    labeler.chain = StubChain()
    suggestions = label_exceptions(cases[0], labeler)
    assert len(sent) == len(cases[0]) and sent[0][0][0] == "system"
    assert json.loads(sent[0][1][1])["exception_type"] in (PAIR, BOOK_ONLY, BANK_ONLY)
    for case, s in zip(cases[0], suggestions):
        possible = case.kind == BOOK_ONLY and case.money_in
        assert s.category == ("DEPOSIT_IN_TRANSIT" if possible else NEEDS_REVIEW)


def test_hybrid_sends_only_bank_only_lines_to_the_llm(cases):
    sent = []

    class StubLLM:
        name = "stub:model"

        def label_all(self, batch):
            sent.extend(batch)
            return [label("INTEREST_CREDIT" if c.money_in else "BANK_CHARGE") for c in batch]

    hybrid, rules = HybridLabeler(StubLLM()), RulesLabeler()
    labels = hybrid.label_all(cases[0])
    assert hybrid.name == "hybrid:stub:model"
    assert sent and [c.exception_id for c in sent] == [c.exception_id for c in cases[0] if c.kind == BANK_ONLY]
    for case, got in zip(cases[0], labels):
        if case.kind == BANK_ONLY:
            assert got.category == ("INTEREST_CREDIT" if case.money_in else "BANK_CHARGE")
        else:
            assert got == rules.label(case)


def test_hybrid_llm_failures_only_touch_bank_only_lines(cases):
    class DownLLM:
        name = "down"

        def label_all(self, batch):
            return [ConnectionError("Ollama is not running") for _ in batch]

    for case, s in zip(cases[0], label_exceptions(cases[0], HybridLabeler(DownLLM()))):
        failed = s.guardrail.startswith("labeler failed")
        assert failed == (case.kind == BANK_ONLY)
        if failed:
            assert s.category == NEEDS_REVIEW


def test_hybrid_refuses_a_short_answer_from_the_llm(cases):
    class ShortLLM:
        name = "short"

        def label_all(self, batch):
            return [label("BANK_CHARGE")]

    with pytest.raises(RuntimeError):
        HybridLabeler(ShortLLM()).label_all(cases[0])
