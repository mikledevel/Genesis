import pytest


def test_pilot_seed_requires_explicit_enablement(monkeypatch):
    from scripts.seed_pilot import seed
    monkeypatch.delenv('GENESIS_PILOT_ENABLED', raising=False)
    with pytest.raises(RuntimeError, match='disabled'):
        seed()


def test_pilot_seed_is_idempotent_and_does_not_reset_password(tmp_path, monkeypatch):
    from scripts.seed_pilot import seed
    from app.db.database import GenesisDB
    from app.core.auth import verify_password
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GENESIS_PILOT_ENABLED', 'true')
    monkeypatch.setenv('PILOT_EMAIL', 'pilot@example.test')
    monkeypatch.setenv('PILOT_PASSWORD', 'test-pilot-password-at-least-24-chars')
    uid = seed()
    db = GenesisDB()
    assert db.is_email_verified(uid)
    assert db.get_user_plan(uid) == 'pro'
    monkeypatch.setenv('PILOT_PASSWORD', 'different-test-password-at-least-24-chars')
    assert seed() == uid
    assert verify_password('test-pilot-password-at-least-24-chars', db.get_user_by_email('pilot@example.test')['password_hash'])


def test_staging_registration_is_closed_before_database_or_email(monkeypatch):
    from app.config import settings
    from app.routers.auth import auth_register
    from fastapi import HTTPException
    monkeypatch.setattr(settings, 'genesis_pilot_enabled', True)
    with pytest.raises(HTTPException) as error:
        auth_register(None, None)
    assert error.value.status_code == 403
