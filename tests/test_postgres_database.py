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


def test_crypto_invoices_and_grants_are_atomic_in_postgres(pg):
    import time
    address='TLjsykFxDXnM4DH1cBT8nYo6MzY81MYPmT'
    ids=[user(pg) for _ in range(6)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        invoices=list(workers.map(lambda uid:pg.create_crypto_invoice(uid,'pro',19,address),ids))
    assert len({i['amount_units'] for i in invoices})==6
    i=invoices[0]
    proof={'txid':'c'*64,'recipient':address,'amount_units':i['amount_units'],
           'timestamp_ms':(i['created_at']+1)*1000,'block_number':87800000}
    with ThreadPoolExecutor(max_workers=4) as workers:
        result=list(workers.map(lambda _:pg.apply_crypto_payment(i['id'],ids[0],proof),range(8)))
    assert sum(not r['already_processed'] for r in result)==1
    assert pg.get_user_plan(ids[0])=='pro'
    until=pg.get_crypto_access(ids[0])['access_until']
    assert int(time.time())+29*86400 < until <= int(time.time())+30*86400+1
    assert GenesisDB(database_url=pg.database_url).get_crypto_invoice(i['id'],ids[0])['status']=='paid'


def test_automatic_check_is_single_use_across_concurrent_workers(pg,monkeypatch):
    from app.core import crypto_payments as crypto
    uid=user(pg);address='TLjsykFxDXnM4DH1cBT8nYo6MzY81MYPmT'
    invoice=pg.create_crypto_invoice(uid,'pro',19,address)
    calls=[]
    def discover(i):
        calls.append(i['id'])
        return {'txid':'e'*64,'recipient':address,'amount_units':i['amount_units'],
                'timestamp_ms':(i['created_at']+1)*1000,'block_number':87800000}
    monkeypatch.setattr(crypto,'discover_transfer',discover)
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda _:crypto.check_invoice_automatically(pg,invoice),range(8)))
    assert len(calls)==1 and pg.get_user_plan(uid)=='pro'
    assert pg.pending_crypto_invoices()==[]


def test_usage_reservation_replay_refund_and_key_quota(pg):
    from app.db.reliability import UsageRejected
    buyer, author = user(pg), user(pg)
    pg.add_transaction(buyer,'seed',100)
    key = pg.create_api_key(buyer,calls_limit=1)['api_key']
    first = pg.reserve_usage(buyer,10,author_id=author,api_key=key,operation_id='reserved')
    with pytest.raises(UsageRejected):
        pg.reserve_usage(buyer,10,author_id=author,api_key=key)
    assert pg.get_balance(buyer)==90 and pg.get_balance(author)==0
    pg.finish_usage(first['id'],refund=True)
    assert pg.get_balance(buyer)==100 and pg.validate_api_key(key)['calls_used']==0
    pg.reserve_usage(buyer,10,author_id=author,operation_id='paid')
    pg.finish_usage('paid',result={'ok':True})
    assert pg.reserve_usage(buyer,10,operation_id='paid')['replayed']
    assert pg.get_balance(author)==8


def test_atomic_monthly_token_and_training_reservations(pg):
    from app.db.reliability import UsageRejected
    uid=user(pg)
    with ThreadPoolExecutor(max_workers=4) as workers:
        result=list(workers.map(lambda _:pg.claim_monthly_use(uid,'server_job',1),range(8)))
    assert sum(result)==1
    rid=pg.reserve_tokens(uid,100,100)
    with pytest.raises(UsageRejected): pg.reserve_tokens(uid,1,100)
    pg.settle_tokens(rid,40)
    pg.reserve_tokens(uid,60,100)


def test_artifact_cache_can_be_lost_and_restored(pg,tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    uid=user(pg)
    folder=tmp_path/'datasets'/'uploads'/uid
    folder.mkdir(parents=True)
    path=folder/'private.csv'
    data=b'x,y\n1,2\n'
    path.write_bytes(data)
    dataset_id=pg.register_dataset(uid,str(path),len(data))
    pg.persist_artifact(path,uid)
    path.unlink()
    assert pg.restore_artifact(path)
    assert path.read_bytes()==data
    assert pg.dataset_path(uid,dataset_id)
    assert pg.dataset_path('someone-else',dataset_id) is None


def test_document_updates_are_serialized_across_instances(pg):
    from app.db.document_store import DatabaseDocument
    one=DatabaseDocument('concurrent',{'items':[]},db=pg)
    two=DatabaseDocument('concurrent',{'items':[]},db=pg)
    def append(i):
        document=one if i%2 else two
        with document.transaction() as current: current['items'].append(i)
    with ThreadPoolExecutor(max_workers=4) as workers: list(workers.map(append,range(20)))
    assert sorted(one.read()['items'])==list(range(20))


def test_job_claims_cancel_and_owner_scope(pg):
    uid=user(pg)
    jid=pg.enqueue_job(uid,'retrain',{'model_id':'one'})
    assert pg.get_job('another-user',jid) is None
    with ThreadPoolExecutor(max_workers=4) as workers: claims=list(workers.map(lambda _:pg.claim_job(),range(4)))
    assert sum(job is not None for job in claims)==1
    assert pg.cancel_job(uid,jid)
    pg.finish_job(jid,result={'unexpected':True})
    assert pg.get_job(uid,jid)['status']=='cancelled'


def test_trace_storage_and_job_attempt_fencing(pg):
    from app.core.tracing import trace_scope
    uid = user(pg)
    with trace_scope("integration", uid, pg) as trace:
        pass
    assert pg.get_trace(uid, trace["id"])["operation"] == "integration"
    assert pg.get_trace("other", trace["id"]) is None
    job_id = pg.enqueue_job(uid, "train", {"path": "datasets/test.csv"})
    job = pg.claim_job()
    pg.set_job_progress(job_id, {"label": "training"}, attempt=job["attempt"])
    pg.finish_job(job_id, result={"stale": True}, attempt=job["attempt"] - 1)
    assert pg.get_job(uid, job_id)["status"] == "running"
    pg.finish_job(job_id, result={"ok": True}, attempt=job["attempt"])
    assert pg.get_job(uid, job_id)["status"] == "succeeded"


def test_backup_restores_accounts_chats_and_artifacts(pg,tmp_path,monkeypatch):
    import os,shutil,subprocess,sys
    from urllib.parse import urlsplit,urlunsplit
    from psycopg import sql
    from app.db.postgres import psycopg,_POOLS,_READY
    if not shutil.which('pg_dump') or not shutil.which('pg_restore'):
        pytest.skip('Install PostgreSQL backup tools for the restoration check')
    uid=user(pg)
    pg.add_transaction(uid,'seed',500)
    pg.create_chat('restore-chat','Restore chat',uid)
    pg.add_message('restore-chat','user','Persist this message')
    root=os.path.dirname(os.path.dirname(__file__))
    monkeypatch.chdir(tmp_path)
    artifact=tmp_path/'datasets'/'uploads'/uid/'data.csv'
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'x,y\n1,2\n')
    pg.persist_artifact(artifact,uid)
    parts=urlsplit(pg.database_url)
    query=parts.query or 'sslmode=disable'
    backup_url=urlunsplit((parts.scheme,parts.netloc,parts.path,query,''))
    env=dict(os.environ,DATABASE_URL=backup_url,BACKUP_DIR=str(tmp_path/'backups'))
    result=subprocess.run([sys.executable,os.path.join(root,'scripts','backup_db.py')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    dump=next((tmp_path/'backups').glob('*.dump'))
    database_name='genesis_restore_'+uuid.uuid4().hex[:10]
    restore_url=urlunsplit((parts.scheme,parts.netloc,'/'+database_name,query,''))
    with psycopg.connect(backup_url,autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(database_name)))
        try:
            restored=subprocess.run(['pg_restore','--no-owner','--no-privileges','--dbname',restore_url,str(dump)],capture_output=True,text=True)
            assert restored.returncode==0,restored.stderr
            repository=GenesisDB(database_url=restore_url)
            assert repository.get_balance(uid)==500
            assert repository.get_messages('restore-chat')[0]['content']=='Persist this message'
            artifact.unlink()
            assert repository.restore_artifact(artifact)
            assert artifact.read_bytes()==b'x,y\n1,2\n'
        finally:
            pool=_POOLS.pop(restore_url,None)
            if pool: pool.close()
            _READY.discard(restore_url)
            admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(database_name)))
