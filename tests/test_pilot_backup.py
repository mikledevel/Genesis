import gzip
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from fastapi import HTTPException


def test_pilot_backup_is_disabled_by_default(monkeypatch):
    from app.routers.pilot import pilot_backup
    from app.config import settings
    monkeypatch.setattr(settings, 'genesis_pilot_enabled', False)
    with pytest.raises(HTTPException) as error:
        pilot_backup('owner')
    assert error.value.status_code == 404


@pytest.mark.parametrize('email,verified', [('foreign@example.test', True), ('owner@example.test', False)])
def test_pilot_backup_requires_the_verified_owner(monkeypatch, email, verified):
    from app.routers.pilot import pilot_backup
    from app.config import settings
    monkeypatch.setattr(settings, 'genesis_pilot_enabled', True)
    monkeypatch.setattr(settings, 'pilot_email', 'owner@example.test')
    db = SimpleNamespace(get_user_by_id=lambda _: {'email': email}, is_email_verified=lambda _: verified)
    monkeypatch.setattr('app.routers.pilot.GenesisDB', lambda: db)
    with pytest.raises(HTTPException) as error:
        pilot_backup('owner')
    assert error.value.status_code == 403


def test_pilot_backup_response_is_private_and_compressed(monkeypatch):
    from app.routers.pilot import pilot_backup
    from app.config import settings
    monkeypatch.setattr(settings, 'genesis_pilot_enabled', True)
    monkeypatch.setattr(settings, 'pilot_email', 'owner@example.test')
    db = SimpleNamespace(get_user_by_id=lambda _: {'email': 'owner@example.test'}, is_email_verified=lambda _: True, is_postgres=True)
    monkeypatch.setattr('app.routers.pilot.GenesisDB', lambda: db)
    monkeypatch.setattr('app.core.pilot_snapshot.snapshot_bytes', lambda _: b'{"snapshot":true}')
    response = pilot_backup('owner')
    assert gzip.decompress(response.body) == b'{"snapshot":true}'
    assert response.headers['cache-control'] == 'no-store'


def test_restore_refuses_non_loopback_database():
    from scripts.restore_pilot_snapshot import restore
    with pytest.raises(ValueError, match='loopback'):
        restore(None, 'postgresql://example.com/production')
