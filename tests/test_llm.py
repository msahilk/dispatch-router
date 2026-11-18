import pytest

from dispatch_router import llm
from dispatch_router.llm import CallSpec, Ledger, ResponseCache, cost_usd


def test_cache_key_stable_and_model_specific():
    a = CallSpec("haiku", "hello", 100)
    assert a.key == CallSpec("haiku", "hello", 100).key
    assert a.key != CallSpec("sonnet", "hello", 100).key
    assert a.key != CallSpec("haiku", "hello", 101).key
    assert len(a.key) == 64  # valid batch custom_id


def test_params_have_no_sampling_args_and_same_template():
    h, s = CallSpec("haiku", "q", 10).params(), CallSpec("sonnet", "q", 10).params()
    for p in (h, s):
        assert "temperature" not in p and "top_p" not in p
    assert h["messages"] == s["messages"] and h["max_tokens"] == s["max_tokens"]


def test_cost_math():
    # Haiku: $0.10 in / $0.50 out per MTok; batch halves it.
    assert cost_usd("haiku", 1_000_000, 1_000_000, batch=False) == pytest.approx(0.60)
    assert cost_usd("haiku", 1_000_000, 1_000_000, batch=True) == pytest.approx(0.30)
    assert cost_usd("sonnet", 1_000_000, 0, batch=False) == pytest.approx(2.00)


def test_cache_roundtrip(tmp_path):
    f = tmp_path / "c.jsonl"
    spec = CallSpec("haiku", "x", 10)
    c = ResponseCache(f)
    assert c.missing([spec, spec]) == [spec]
    c.put({"key": spec.key, "text": "hi"})
    assert ResponseCache(f).get(spec)["text"] == "hi"
    assert ResponseCache(f).missing([spec]) == []


def test_ledger_and_budget_guard(tmp_path, monkeypatch):
    led = Ledger(tmp_path / "l.csv")
    assert led.total() == 0
    led.append("p", "m", "batch", 3, 10, 10, 4.9)
    assert led.total() == pytest.approx(4.9)
    monkeypatch.setattr(llm, "Ledger", lambda: led)
    est = llm.estimate([CallSpec("sonnet", "x" * 4000, 1000, "gsm8k")] * 200, batch=False)
    with pytest.raises(llm.BudgetExceeded):
        llm.check_budget(est)
