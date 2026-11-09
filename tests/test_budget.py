import pytest

from ar_kernel.budget import Budget, BudgetError, usage_tokens
from ar_kernel.config import KernelConfig

PRICES = {"input": 2.0, "cached_input": 0.5, "output": 8.0}


def test_usage_forms():
    assert usage_tokens({"prompt_tokens": 100, "completion_tokens": 10,
                         "prompt_tokens_details": {"cached_tokens": 60}}) == (40, 60, 10)
    assert usage_tokens({"prompt_tokens": 100, "completion_tokens": 10,
                         "prompt_cache_hit_tokens": 90}) == (10, 90, 10)
    assert usage_tokens({"input_tokens": 50, "output_tokens": 5,
                         "input_tokens_details": {"cached_tokens": 50}}) == (0, 50, 5)
    assert usage_tokens(None) == (0, 0, 0)


def test_cost_and_dollar_cap():
    b = Budget(max_usd=0.001, prices=PRICES)
    cost = b.record(200, {"prompt_tokens": 400, "completion_tokens": 50,
                          "prompt_tokens_details": {"cached_tokens": 200}})
    assert cost == pytest.approx((200 * 2 + 200 * 0.5 + 50 * 8) / 1e6)
    assert b.exhausted() is None
    b.record(200, {"prompt_tokens": 1000, "completion_tokens": 0})
    assert "budget" in b.exhausted()


def test_without_prices_cost_is_unknown_and_nothing_is_capped():
    b = Budget()
    assert b.record(200, {"prompt_tokens": 60, "completion_tokens": 50}) is None
    assert b.exhausted() is None and b.snapshot()["tokens"] == 110 and b.snapshot()["usd"] is None


def test_no_cap_by_default():
    b = Budget.from_config(KernelConfig.load())
    b.record(200, {"prompt_tokens": 10 ** 9, "completion_tokens": 10 ** 9})
    assert b.exhausted() is None


def test_dollar_cap_needs_prices():
    cfg = KernelConfig(raw={"budget": {"max_usd": 5, "usd_per_mtok": {"input": 1}}}, repo_root=None)
    with pytest.raises(BudgetError, match="usd_per_mtok"):
        Budget.from_config(cfg)


def _call(run, usage, mock=False, budget=Budget(prices=PRICES)):
    from ar_kernel.archive.activity import add_call
    add_call(run, node="n1", phase="improve_recipe", attempt=1, role=None, tool=None, conversation_id="c", turn=0,
             transcript=None, status=200, tokens=usage_tokens(usage), cost_usd=None if mock else budget.cost(usage),
             mock=mock, audit_hits=[])


def test_load_counts_real_calls_only_and_twice_counts_once(tmp_path):
    from ar_kernel.archive.db import open_db
    _call(tmp_path, {"prompt_tokens": 10, "completion_tokens": 5})
    _call(tmp_path, {"prompt_tokens": 999, "completion_tokens": 9}, mock=True)
    b = Budget(prices=PRICES)
    b.load(open_db(tmp_path)), b.load(open_db(tmp_path))
    assert b.snapshot() == {"usd": (10 * 2.0 + 5 * 8.0) / 1e6, "tokens": 15, "calls": 1, "max_usd": None}


def test_the_spend_is_the_same_after_a_reopen(tmp_path):
    """What the live budget counted is what a resumed run loads."""
    from ar_kernel.archive.db import open_db
    live = Budget(prices=PRICES)
    for usage in ({"prompt_tokens": 1000, "completion_tokens": 50}, {"input_tokens": 7, "output_tokens": 3,
                                                                       "input_tokens_details": {"cached_tokens": 4}}):
        live.record(200, usage)
        _call(tmp_path, usage)
    resumed = Budget(prices=PRICES)
    resumed.load(open_db(tmp_path))
    assert resumed.snapshot() == live.snapshot()


def test_error_rate_window():
    b = Budget()
    for status in (200, 500, 502, 200, 400, 599, 401, 404, 402, 403, 200, 422):   # 400: the agent's own request; 599: a connection error
        b.record(status, None)
    rate, n = b.error_rate(300)
    assert (rate, n) == (0.5, 12)


def test_string_prices_and_cap_are_numbers():     # YAML loads "2e-6" (no dot) as a string
    b = Budget(max_usd="0.001", prices={"input": "2", "cached_input": "5e-1", "output": "8"})
    assert b.record(200, {"prompt_tokens": 1000, "completion_tokens": 0}) == pytest.approx(0.002)
    assert "budget" in b.exhausted()


@pytest.mark.parametrize("max_usd, prices", [("five", PRICES), (-1, PRICES),
                                             (None, {**PRICES, "output": "x"}), (None, {**PRICES, "input": -2})])
def test_unparsable_or_negative_budget_values_are_rejected(max_usd, prices):
    with pytest.raises(BudgetError):
        Budget(max_usd=max_usd, prices=prices)
