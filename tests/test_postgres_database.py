"""Integration checks against a disposable local PostgreSQL database.

Set GENESIS_TEST_POSTGRES_URL to a loopback-only test database. Tables in its
private genesis schema are truncated between cases; production URLs are refused.
"""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import pytest

from app.db.database import GenesisDB


@pytest.fixture
def pg():
    url = os.environ.get('GENESIS_TEST_POSTGRES_URL')
    if not url:
        pytest.skip('Set GENESIS_TEST_POSTGRES_URL for PostgreSQL integration checks')
    if urlparse(url).hostname not in ('localhost', '127.0.0.1', '::1'):
        pytest.fail('PostgreSQL integration tests require a disposable loopback database')
    db = GenesisDB(database_url=url)
    with db._connect() as conn:
        tables = conn.execute("SELECT quote_ident(tablename) FROM pg_catalog.pg_tables WHERE schemaname='genesis'").fetchall()
        conn.execute('TRUNCATE ' + ','.join('genesis.' + t[0] for t in tables) + ' RESTART IDENTITY CASCADE')
    return db


def user(db):
    return db.create_user(uuid.uuid4().hex + '@example.test', 'hashed', 'Postgres user')['id']


def test_account_tokens_and_balance_survive_new_repository(pg):
    uid = user(pg)
    token = pg.create_auth_token(uid, 'verify_email', 3600)
    pg.add_transaction(uid, 'topup', 1900, stripe_session_id='cs_pg')
    other = GenesisDB(database_url=pg.database_url)
    assert other.get_balance(uid) == 1900
    assert other.count_recent_auth_tokens(uid, 'verify_email') == 1
    assert other.consume_auth_token(token, 'verify_email') == uid
    assert other.consume_auth_token(token, 'verify_email') is None
    assert other.stripe_session_already_processed('cs_pg')


def test_parallel_checkout_reserves_one_attempt(pg):
    uid = user(pg)
    with ThreadPoolExecutor(max_workers=4) as workers:
        result = list(workers.map(lambda i: pg.reserve_stripe_checkout(uid, 'pro', f'attempt_{i}'), range(8)))
    assert sum(r['status'] == 'reserved' for r in result) == 1
    assert len({r['attempt_id'] for r in result}) == 1


def test_parallel_subscription_events_are_atomic_and_idempotent(pg):
    uid = user(pg)
    pg.save_stripe_customer(uid, 'cus_pg')
    def apply(_):
        return pg.apply_stripe_subscription_event('evt_pg', 'customer.subscription.updated', uid,
                                                  'cus_pg', 'sub_pg', 'active', 'price_pg',
                                                  2_000_000_000, False, 'pro')
    with ThreadPoolExecutor(max_workers=4) as workers:
        result = list(workers.map(apply, range(8)))
    assert sum(r['status'] == 'already_processed' for r in result) == 7
    assert pg.get_user_plan(uid) == 'pro'
    pg.apply_stripe_subscription_event('evt_cancel', 'customer.subscription.deleted', uid,
                                       'cus_pg', 'sub_pg', 'canceled', 'price_pg', None, False, 'pro')
    assert pg.get_user_plan(uid) == 'free'


def test_parallel_charges_cannot_spend_same_balance_twice(pg):
    uid = user(pg)
    pg.add_transaction(uid, 'topup', 100)
    with ThreadPoolExecutor(max_workers=2) as workers:
        result = list(workers.map(lambda _: pg.charge_for_usage(uid, 100, 'one use'), range(2)))
    assert sum(r['ok'] for r in result) == 1
    assert pg.get_balance(uid) == 0


def test_messages_ids_activity_and_cascade(pg):
    uid = user(pg)
    pg.create_chat('chat_pg', 'Title', uid)
    pg.create_chat('chat_pg', 'Duplicate ignored', uid)
    mid = pg.add_message('chat_pg', 'user', "50%? 'quoted'", {'test': True})
    assert isinstance(mid, int)
    assert pg.get_messages('chat_pg')[0]['content'] == "50%? 'quoted'"
    assert pg.get_activity_last_7_days(uid)[-1]['count'] == 1
    pg.delete_chat('chat_pg')
    assert pg.get_message_count('chat_pg') == 0


def test_monthly_usage_admin_and_versioned_prompts(pg):
    uid = user(pg)
    pg.record_feature_use(uid, 'train')
    assert pg.count_feature_use_this_month(uid, 'train') == 1
    assert pg.admin_list_users('Postgres')['total'] == 1
    assert pg.admin_overview()['signups_7d'] == 1
    prompt = pg.create_prompt(uid, 'Prompt', 'Version one')
    pg.add_prompt_version(prompt['id'], uid, 'Version two')
    assert pg.get_prompt(prompt['id'], uid)['current_version'] == 2


def test_scheduler_interval_arithmetic_and_claims(pg):
    uid = user(pg)
    job = pg.create_scheduled_job('skill_pg', 'agent_pg', uid, {}, 60)
    assert not pg.claim_due_jobs()
    with pg._connect() as conn:
        conn.execute("UPDATE scheduled_jobs SET next_run_at=datetime('now','-1 minutes') WHERE id=?", (job['id'],))
    assert len(pg.claim_due_jobs()) == 1
    assert not pg.claim_due_jobs()
    pg.complete_scheduled_job_run(job['id'], True, {'ok': True}, None)
    assert not pg.claim_due_jobs()


def test_schema_is_private_and_all_tables_enable_rls(pg):
    with pg._connect() as conn:
        rows = conn.execute("SELECT relrowsecurity FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n "
                            "ON n.oid=c.relnamespace WHERE n.nspname='genesis' AND c.relkind='r'").fetchall()
    assert rows and all(r[0] for r in rows)


def test_sqlite_import_preserves_accounts_balances_and_message_sequences(pg, tmp_path):
    from scripts.import_sqlite_postgres import import_database
    path = tmp_path / 'export.sqlite'
    old = GenesisDB(str(path))
    uid = user(old)
    old.add_transaction(uid, 'topup', 1900, stripe_session_id='cs_import')
    old.create_chat('imported', 'Imported chat', uid)
    original_id = old.add_message('imported', 'user', 'Original')
    counts = import_database(path, pg.database_url)
    assert counts['users'] == 1
    assert pg.get_balance(uid) == 1900
    assert pg.stripe_session_already_processed('cs_import')
    assert pg.add_message('imported', 'assistant', 'New message') > original_id
    assert old.get_message_count('imported') == 1  # source stayed read-only


def test_sqlite_import_refuses_to_overwrite_existing_target(pg, tmp_path):
    from scripts.import_sqlite_postgres import import_database
    uid = user(pg)
    path = tmp_path / 'export.sqlite'
    old = GenesisDB(str(path))
    user(old)
    with pytest.raises(ValueError, match='not empty'):
        import_database(path, pg.database_url)
    assert pg.get_user_by_id(uid) is not None
    assert pg.admin_overview()['users'] == 1
