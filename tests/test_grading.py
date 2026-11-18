import pandas as pd
import pytest

from dispatch_router.datasets import gsm8k_gold, stratified_sample
from dispatch_router.prompts import extract_answer_line, grade, mmlu_prompt


@pytest.mark.parametrize(
    "text,gold,correct",
    [
        ("blah\nANSWER: 72", "72", True),
        ("ANSWER: $1,250.00", "1250", True),
        ("**ANSWER: 18 dollars**", "18", True),
        ("first try ANSWER: 5\nwait, fix\nANSWER: 6", "6", True),  # last answer line wins
        ("answer: -3", "-3", True),
        ("ANSWER: 0.5", "0.50", True),
        ("ANSWER: 71", "72", False),
    ],
)
def test_number_grading(text, gold, correct):
    assert grade("number", gold, text)["correct"] is correct


def test_missing_answer_line_is_parse_fail():
    g = grade("number", "72", "The total is 72.")
    assert g == {"extracted": None, "correct": False, "parse_fail": True}


@pytest.mark.parametrize(
    "text,correct,pf",
    [("ANSWER: B", True, False), ("ANSWER: (b)", True, False), ("ANSWER: B. Paris", True, False),
     ("ANSWER: C", False, False), ("ANSWER: none of these", False, True)],
)
def test_letter_grading(text, correct, pf):
    g = grade("letter", "B", text)
    assert g["correct"] is correct and g["parse_fail"] is pf


def test_exact_grading_strips_backticks_and_whitespace():
    assert grade("exact", "[1, 2]", "ANSWER: `[1, 2]`  ")["correct"]
    assert not grade("exact", "[1, 2]", "ANSWER: [1,2]")["correct"]


def test_extract_none_when_empty():
    assert extract_answer_line("ANSWER:   ") is None


def test_gsm8k_gold():
    assert gsm8k_gold("work...\n#### 1,000") == "1000"


def test_mmlu_prompt_lists_options():
    p = mmlu_prompt("Q?", "high_school_physics", ["w", "x", "y", "z"])
    assert "A. w\nB. x\nC. y\nD. z" in p and "high school physics" in p


def test_stratified_sample_keeps_shares():
    df = pd.DataFrame({"g": ["a"] * 80 + ["b"] * 20, "v": range(100)})
    s = stratified_sample(df, 50, "g", seed=0)
    assert len(s) == 50 and (s.g == "a").sum() == 40
    assert s.equals(stratified_sample(df, 50, "g", seed=0))  # seeded
