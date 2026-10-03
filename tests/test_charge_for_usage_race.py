"""Tests for GenesisDB.charge_for_usage - in particular the double-spend race fix.

BEFORE the fix, charge_for_usage() read the balance (get_balance) and deducted it
(add_transaction) as two SEPARATE transactions. Two concurrent calls for the same buyer could
both read the same starting balance, both see it as sufficient, and both deduct - letting the
balance go negative. This is exercised for real here with actual OS threads hitting a real
(temp-file) SQLite database - a mock can't catch a race condition, only real concurrent
execution can - and it needs no external dependencies (sqlite3 + threading are both stdlib),
so it runs even in environments where the rest of the app's dependencies aren't installed.
"""
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.database import GenesisDB


def _fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # GenesisDB._init_db creates the schema fresh
    return GenesisDB(db_path=path), path


def test_charge_for_usage_never_goes_negative_under_concurrency():
    db, path = _fresh_db()
    try:
        user_id = db.create_user("racer@example.com", "hash")["id"]
        # Give this user exactly enough balance for ONE $1.00 charge.
        db.add_transaction(user_id, "test_topup", 100, "seed balance")
        assert db.get_balance(user_id) == 100

        # Fire 20 concurrent attempts to charge $1.00 each. At most ONE can legitimately
        # succeed - the other 19 must see insufficient_balance, never a negative balance.
        results = []
        results_lock = threading.Lock()

        def _attempt():
            r = db.charge_for_usage(user_id, 100, "concurrent charge attempt")
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=_attempt) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        successes = [r for r in results if r["ok"]]
        failures = [r for r in results if not r["ok"]]
        final_balance = db.get_balance(user_id)

        assert len(results) == 20, "every thread should have completed"
        assert len(successes) == 1, f"expected exactly 1 successful charge, got {len(successes)}: {results}"
        assert len(failures) == 19
        assert all(r["reason"] == "insufficient_balance" for r in failures)
        assert final_balance == 0, f"balance went to {final_balance}, expected exactly 0 (never negative)"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_charge_for_usage_credits_author_share_correctly():
    db, path = _fresh_db()
    try:
        buyer = db.create_user("buyer@example.com", "hash")["id"]
        author = db.create_user("author@example.com", "hash")["id"]
        db.add_transaction(buyer, "test_topup", 1000, "seed")

        result = db.charge_for_usage(buyer, 100, "used a paid model", author_user_id=author,
                                      platform_fee_pct=25.0)
        assert result == {"ok": True, "charged_cents": 100}
        assert db.get_balance(buyer) == 900
        # 100 cents - 25% platform fee = 75 cents to the author.
        assert db.get_balance(author) == 75

        txns = db.get_transactions(author)
        assert len(txns) == 1
        assert txns[0]["type"] == "usage_earning"
        assert txns[0]["amount_cents"] == 75
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_charge_for_usage_insufficient_balance_does_not_charge_anything():
    db, path = _fresh_db()
    try:
        buyer = db.create_user("buyer2@example.com", "hash")["id"]
        author = db.create_user("author2@example.com", "hash")["id"]
        db.add_transaction(buyer, "test_topup", 50, "seed")

        result = db.charge_for_usage(buyer, 100, "too expensive", author_user_id=author)
        assert result["ok"] is False
        assert result["reason"] == "insufficient_balance"
        assert result["balance_cents"] == 50
        assert result["needed_cents"] == 100
        # Nothing should have moved beyond the seed top-up itself - no charge, no author
        # credit, no new ledger rows for either party.
        assert db.get_balance(buyer) == 50
        assert db.get_balance(author) == 0
        assert [t["type"] for t in db.get_transactions(buyer)] == ["test_topup"]
        assert db.get_transactions(author) == []
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_use_api_key_never_exceeds_calls_limit_under_concurrency():
    """Same double-spend shape as charge_for_usage, on API key call quotas instead of money -
    see the BUGFIX note on GenesisDB.use_api_key."""
    db, path = _fresh_db()
    try:
        owner = db.create_user("keyowner@example.com", "hash")["id"]
        key = db.create_api_key(owner, "test key", calls_limit=1)["api_key"]

        results = []
        results_lock = threading.Lock()

        def _attempt():
            r = db.use_api_key(key, "some-model")
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=_attempt) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 20
        assert sum(1 for r in results if r is True) == 1, f"expected exactly 1 success, got {results}"
        final = db.validate_api_key(key)
        assert final["calls_used"] == 1, f"calls_used ended at {final['calls_used']}, expected exactly 1"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


if __name__ == "__main__":
    test_charge_for_usage_never_goes_negative_under_concurrency()
    print("PASS: test_charge_for_usage_never_goes_negative_under_concurrency")
    test_charge_for_usage_credits_author_share_correctly()
    print("PASS: test_charge_for_usage_credits_author_share_correctly")
    test_charge_for_usage_insufficient_balance_does_not_charge_anything()
    print("PASS: test_charge_for_usage_insufficient_balance_does_not_charge_anything")
    test_use_api_key_never_exceeds_calls_limit_under_concurrency()
    print("PASS: test_use_api_key_never_exceeds_calls_limit_under_concurrency")
    print("\nALL PASSED")
