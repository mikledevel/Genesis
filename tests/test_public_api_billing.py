"""Tests for app/core/public_api_billing.py (pure pricing math) and the reconcile_charge /
refund flow against a REAL GenesisDB (temp-file SQLite, no mocking needed - same approach as
tests/test_charge_for_usage_race.py, since money logic deserves a real database, not a mock).
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pydantic_settings  # noqa
except ImportError:
    fake_ps = types.ModuleType("pydantic_settings")
    class _BaseSettings:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
    fake_ps.BaseSettings = _BaseSettings
    fake_ps.SettingsConfigDict = lambda **kw: kw
    sys.modules["pydantic_settings"] = fake_ps

from app.config import settings
from app.core import public_api_billing as billing
from app.db.database import GenesisDB


def setup_module(module):
    settings.public_api_input_price_cents_per_million = 50
    settings.public_api_output_price_cents_per_million = 200
    settings.public_api_min_charge_cents = 1


def _fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return GenesisDB(db_path=path), path


# ---------------------------------------------------------------- compute_cost_cents ----

def test_compute_cost_cents_basic_math():
    # 1,000,000 input tokens @ 50c/M = 50c; 1,000,000 output tokens @ 200c/M = 200c
    assert billing.compute_cost_cents(1_000_000, 0) == 50
    assert billing.compute_cost_cents(0, 1_000_000) == 200
    assert billing.compute_cost_cents(1_000_000, 1_000_000) == 250


def test_compute_cost_cents_rounds_up_not_down():
    # A tiny request: 100 input + 50 output tokens costs a small fraction of a cent -
    # must round UP to the minimum charge, never down to 0.
    cost = billing.compute_cost_cents(100, 50)
    assert cost == settings.public_api_min_charge_cents == 1


def test_compute_cost_cents_never_below_minimum_even_for_zero_tokens():
    assert billing.compute_cost_cents(0, 0) == settings.public_api_min_charge_cents


def test_compute_cost_cents_ceils_a_fractional_result():
    # 10,001 input tokens @ 50c/M = 0.50005 cents -> ceil -> 1 cent (not 0, not 0.5)
    assert billing.compute_cost_cents(10_001, 0) == 1
    # A bigger example landing just over a whole-cent boundary: 40,001 tokens @ 50c/M =
    # 2.00005 cents -> ceil -> 3 cents (ceil rounds UP past the boundary, not to it)
    assert billing.compute_cost_cents(40_001, 0) == 3


# ---------------------------------------------------------------- estimate_prompt_tokens ----

def test_estimate_prompt_tokens_roughly_4_chars_per_token():
    messages = [{"role": "user", "content": "a" * 400}]
    assert billing.estimate_prompt_tokens(messages) == 100


def test_estimate_prompt_tokens_sums_all_messages():
    messages = [{"role": "system", "content": "a" * 40}, {"role": "user", "content": "b" * 60}]
    assert billing.estimate_prompt_tokens(messages) == 25  # ceil(100/4)


def test_estimate_prompt_tokens_never_zero():
    assert billing.estimate_prompt_tokens([]) == 1
    assert billing.estimate_prompt_tokens([{"role": "user", "content": ""}]) == 1


def test_estimate_worst_case_cost_uses_max_tokens_as_completion_ceiling():
    messages = [{"role": "user", "content": "a" * 4000}]  # -> 1000 estimated prompt tokens
    cost = billing.estimate_worst_case_cost_cents(messages, max_tokens=2000)
    assert cost == billing.compute_cost_cents(1000, 2000)


# ---------------------------------------------------------------- reconcile_charge ----

def test_reconcile_refunds_the_difference_when_actual_usage_is_under_the_estimate():
    db, path = _fresh_db()
    try:
        user = db.create_user("dev1@example.com", "hash")["id"]
        db.add_transaction(user, "test_topup", 1000, "seed")

        pre_charged = 100  # pretend we pre-charged $1.00 for a worst-case max_tokens
        db.charge_for_usage(user, pre_charged, "pre-charge", related_id="gen-testkey")
        assert db.get_balance(user) == 900

        # Actual usage turned out much cheaper: real cost is 1 cent (the minimum).
        final = billing.reconcile_charge(db, user, "gen-testkey", pre_charged,
                                          prompt_tokens=10, completion_tokens=10)
        assert final == 1
        assert db.get_balance(user) == 999  # 900 + (100 - 1) refunded
        refund_txns = [t for t in db.get_transactions(user) if t["type"] == "public_api_refund"]
        assert len(refund_txns) == 1
        assert refund_txns[0]["amount_cents"] == 99
    finally:
        os.unlink(path)


def test_reconcile_charges_the_shortfall_when_actual_usage_exceeds_the_estimate():
    db, path = _fresh_db()
    try:
        user = db.create_user("dev2@example.com", "hash")["id"]
        db.add_transaction(user, "test_topup", 1000, "seed")

        pre_charged = 1  # pretend the prompt-token estimate undershot badly
        db.charge_for_usage(user, pre_charged, "pre-charge", related_id="gen-testkey")
        assert db.get_balance(user) == 999

        # Real usage costs more than what was pre-charged.
        real_cost = billing.compute_cost_cents(2_000_000, 0)  # 100 cents
        final = billing.reconcile_charge(db, user, "gen-testkey", pre_charged,
                                          prompt_tokens=2_000_000, completion_tokens=0)
        assert final == real_cost == 100
        assert db.get_balance(user) == 999 - (real_cost - pre_charged)
    finally:
        os.unlink(path)


def test_reconcile_shortfall_that_cannot_be_collected_does_not_raise():
    """If the caller's balance dropped to near-zero between the pre-charge and the real
    completion coming back, the extra charge attempt fails - reconcile_charge must not raise
    or crash the response; it eats the shortfall (see its docstring)."""
    db, path = _fresh_db()
    try:
        user = db.create_user("dev3@example.com", "hash")["id"]
        db.add_transaction(user, "test_topup", 1, "seed")  # only 1 cent, no room for a top-up charge

        pre_charged = 1
        db.charge_for_usage(user, pre_charged, "pre-charge", related_id="gen-testkey")
        assert db.get_balance(user) == 0

        final = billing.reconcile_charge(db, user, "gen-testkey", pre_charged,
                                          prompt_tokens=2_000_000, completion_tokens=0)
        # Returned amount reflects what was ACTUALLY collected (just the pre-charge) - not
        # the full real cost, since the extra charge attempt failed.
        assert final == pre_charged == 1
        assert db.get_balance(user) == 0  # never went negative
    finally:
        os.unlink(path)


def test_reconcile_exact_match_neither_refunds_nor_charges():
    db, path = _fresh_db()
    try:
        user = db.create_user("dev4@example.com", "hash")["id"]
        db.add_transaction(user, "test_topup", 1000, "seed")
        pre_charged = billing.compute_cost_cents(100, 100)
        db.charge_for_usage(user, pre_charged, "pre-charge", related_id="gen-testkey")
        balance_after_precharge = db.get_balance(user)

        final = billing.reconcile_charge(db, user, "gen-testkey", pre_charged,
                                          prompt_tokens=100, completion_tokens=100)
        assert final == pre_charged
        assert db.get_balance(user) == balance_after_precharge  # unchanged
    finally:
        os.unlink(path)


if __name__ == "__main__":
    setup_module(None)
    passed, failed = 0, 0
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"FAIL  {name}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
