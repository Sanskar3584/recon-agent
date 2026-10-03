"""Segment 3 tests: the scorer rewards the right answer and catches wrong or missing matches."""
import copy

import pytest

from recon import generator
from recon.loader import load_bank, load_book
from recon.matcher import MATCHED, NEEDS_CORRECTION, Match, MatchResult, match
from recon.score import MATCH_CATEGORIES, load_answer_key, score_matching


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("data")
    generator.write_outputs(generator.generate(generator.GeneratorConfig(seed=42)), folder)
    books, banks = load_book(folder / "cash_book.csv"), load_bank(folder / "bank_statement.csv")
    return books, banks, load_answer_key(folder / "answer_key.csv")


def perfect_result(books, banks, key):
    """Build the result a flawless matcher would produce, straight from the answer key."""
    matches, used_book, used_bank = [], set(), set()
    for item in key:
        if item["category"] in MATCH_CATEGORIES or item["category"] == "AMOUNT_ERROR":
            status = MATCHED if item["category"] in MATCH_CATEGORIES else NEEDS_CORRECTION
            matches.append(Match(f"M-{len(matches)}", "key", status, item["book_ids"], item["bank_ids"], 0, 0, 100, ""))
            used_book.update(item["book_ids"])
            used_bank.update(item["bank_ids"])
    return MatchResult(matches, [l for l in books if l.book_id not in used_book],
                       [l for l in banks if l.bank_id not in used_bank], None)


def test_a_perfect_result_scores_100(run):
    scores = score_matching(perfect_result(*run), run[2])
    assert scores["match_precision_pct"] == scores["auto_match_rate_pct"] == scores["exceptions_surfaced_pct"] == 100.0


def test_a_wrong_pair_costs_precision_and_recall(run):
    result = perfect_result(*run)
    first, second = [m for m in result.matches if m.status == MATCHED][:2]
    first.bank_ids, second.bank_ids = second.bank_ids, first.bank_ids     # swap two matches
    scores = score_matching(result, run[2])
    assert scores["match_precision_pct"] < 100 and scores["auto_match_rate_pct"] < 100


def test_a_missed_match_costs_recall_but_not_precision(run):
    books, banks, key = run
    result = perfect_result(books, banks, key)
    dropped = next(m for m in result.matches if m.status == MATCHED)
    result.matches.remove(dropped)
    result.unmatched_book += [l for l in books if l.book_id in dropped.book_ids]
    result.unmatched_bank += [l for l in banks if l.bank_id in dropped.bank_ids]
    scores = score_matching(result, key)
    assert scores["match_precision_pct"] == 100 and scores["auto_match_rate_pct"] < 100


def test_hiding_an_exception_inside_a_match_is_caught(run):
    books, banks, key = run
    result = perfect_result(books, banks, key)
    charge = next(i for i in key if i["category"] == "BANK_CHARGE")
    duplicate = next(i for i in key if i["category"] == "DUPLICATE_BOOK")
    result.matches.append(Match("M-X", "bad", MATCHED, duplicate["book_ids"], charge["bank_ids"], 0, 0, 50, ""))
    result.unmatched_book = [l for l in result.unmatched_book if l.book_id not in duplicate["book_ids"]]
    result.unmatched_bank = [l for l in result.unmatched_bank if l.bank_id not in charge["bank_ids"]]
    scores = score_matching(result, key)
    assert scores["exceptions_surfaced_pct"] < 100 and scores["match_precision_pct"] < 100


def test_the_real_matcher_on_seed_42(run):
    books, banks, key = run
    scores = score_matching(match(books, banks), key)
    assert scores["match_precision_pct"] == 100.0
    assert scores["auto_match_rate_pct"] == 100.0
    assert scores["exceptions_surfaced_pct"] == 100.0
