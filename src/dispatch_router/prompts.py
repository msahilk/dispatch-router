"""Prompt templates, answer extraction and grading.

Every prompt asks for brief reasoning and a final line `ANSWER: <x>`. The same
template is sent to Haiku and Sonnet. Grading depends on the item's answer_type:

  number  -> numeric equality after stripping $, commas, %, units (GSM8K, math)
  letter  -> exact letter A-D (MMLU)
  exact   -> exact string after stripping surrounding whitespace/quotes/backticks
             (Python "what does this print?", SQL single values)
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any

LETTERS = "ABCD"

GSM8K_TEMPLATE = """Solve the following math word problem. Reason briefly, step by step. \
Then write the final answer on its own last line in exactly this format:
ANSWER: <number>
Use digits only on the ANSWER line (no units, no commas).

Problem: {question}"""

MMLU_TEMPLATE = """Answer the following multiple-choice question about {subject}. \
Reason briefly. Then write the final answer on its own last line in exactly this format:
ANSWER: <letter>
where <letter> is one of A, B, C, D.

Question: {question}
{options}"""


def gsm8k_prompt(question: str) -> str:
    return GSM8K_TEMPLATE.format(question=question.strip())


def mmlu_prompt(question: str, subject: str, choices: list[str]) -> str:
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return MMLU_TEMPLATE.format(subject=subject.replace("_", " "), question=question.strip(), options=options)


# ------------------------------------------------------------------ extraction

_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.*)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_answer_line(text: str) -> str | None:
    """Content after the LAST `ANSWER:` marker, with markdown decoration removed.
    Returns None if the model never wrote an answer line (a parse_fail)."""
    matches = _ANSWER_RE.findall(text or "")
    if not matches:
        return None
    raw = matches[-1].strip()
    raw = raw.strip("*_ ").strip()
    if raw.startswith("`") and raw.endswith("`") and len(raw) >= 2:
        raw = raw.strip("`")
    return raw.strip() or None


def parse_number(s: str) -> Decimal | None:
    s = s.replace(",", "").replace("$", "").replace("\\", "")
    # "\frac" / "\boxed{...}" rarely appear on the ANSWER line, but be lenient about braces
    s = s.replace("{", " ").replace("}", " ")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return Decimal(m.group(0))
    except InvalidOperation:
        return None


def parse_letter(s: str) -> str | None:
    m = re.match(r"^\(?([A-Da-d])\)?(?:[.):\s]|$)", s.strip())
    return m.group(1).upper() if m else None


def normalize_exact(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "`":
        s = s[1:-1]
    return s.rstrip()


# --------------------------------------------------------------------- grading


def numbers_equal(a: Decimal, b: Decimal, rel_tol: float = 1e-6) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=1e-9)


def grade(answer_type: str, gold: str, text: str) -> dict[str, Any]:
    """Grade a model response. Returns extracted answer, correct, parse_fail."""
    line = extract_answer_line(text)
    if line is None:
        return {"extracted": None, "correct": False, "parse_fail": True}
    if answer_type == "number":
        got = parse_number(line)
        want = parse_number(gold)
        if got is None or want is None:
            return {"extracted": line, "correct": False, "parse_fail": got is None}
        return {"extracted": str(got), "correct": numbers_equal(got, want), "parse_fail": False}
    if answer_type == "letter":
        got = parse_letter(line)
        return {"extracted": got, "correct": got == gold.upper(), "parse_fail": got is None}
    if answer_type == "exact":
        got = normalize_exact(line)
        return {"extracted": got, "correct": got == normalize_exact(gold), "parse_fail": False}
    raise ValueError(f"unknown answer_type {answer_type!r}")
