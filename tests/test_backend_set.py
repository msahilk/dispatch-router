import pytest

from dispatch_router.backend_set import (
    NearTie, build_backend_set, round_safe, run_python, run_sql,
)
from dispatch_router.prompts import grade


@pytest.fixture(scope="module")
def built():
    return build_backend_set(42)


def test_python_verifier_executes():
    assert run_python("print(-7 // 2)") == "-4"
    with pytest.raises(ValueError):
        run_python("print(1)\nprint(2)")  # must be one line
    with pytest.raises(RuntimeError):
        run_python("raise SystemExit(1)")


def test_sql_verifier():
    assert run_sql("SELECT COUNT(*) FROM users WHERE country != 'US';") == ("3", "number")
    assert run_sql("SELECT COUNT(*) FROM users WHERE country NOT IN "
                   "(SELECT country FROM users WHERE age > 40);") == ("0", "number")
    assert run_sql("SELECT name FROM users WHERE id = 5;") == ("erin", "exact")
    with pytest.raises(ValueError):
        run_sql("SELECT id FROM users;")


def test_round_safe_refuses_ties():
    assert str(round_safe(8.789, 2)) == "8.79"
    with pytest.raises(NearTie):
        round_safe(1.005, 2)


def test_build_is_deterministic_and_balanced(built):
    items, _ = built
    again, _ = build_backend_set(42)
    assert [i["gold"] for i in items] == [i["gold"] for i in again]
    assert len(items) >= 150
    cats = {i["category"] for i in items}
    assert cats == {"python", "sql", "infra", "finance"}
    assert len({i["id"] for i in items}) == len(items)
    assert len({i["prompt"] for i in items}) == len(items)


def test_every_gold_grades_as_correct(built):
    """A model that answers exactly the gold must be graded correct."""
    for it in built[0]:
        assert grade(it["answer_type"], it["gold"], f"ANSWER: {it['gold']}")["correct"], it["id"]
