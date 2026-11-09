"""Account settings persist on the server and never mutate another user's account."""
import time
import pytest
import app.main
from fastapi.testclient import TestClient
from app.db.database import GenesisDB
from app.core.auth import create_token, hash_password, verify_password

@pytest.fixture
def account(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = GenesisDB()
    user = db.create_user('settings@example.test', hash_password('old-password'), 'Before')
    other = db.create_user('other@example.test', hash_password('other-password'), 'Other')
    client = TestClient(app.main.app)
    client.headers['x-forwarded-for'] = '10.42.0.' + str(account.counter)
    account.counter += 1
    client.headers['Authorization'] = 'Bearer ' + create_token(user['id'])
    return client, db, user, other
account.counter = 1

def test_profile_persists_and_existing_session_still_works(account):
    client, db, user, other = account
    response = client.patch('/api/auth/profile', json={'name': '  After  '})
    assert response.status_code == 200
    assert response.json()['user']['name'] == 'After'
    assert 'password_hash' not in response.json()['user']
    assert db.get_user_by_id(user['id'])['name'] == 'After'
    assert db.get_user_by_id(other['id'])['name'] == 'Other'
    assert client.get('/api/auth/me').status_code == 200

@pytest.mark.parametrize('body', [{'name':''}, {'name':' '}, {'name':'x'*101}, {'name':123}, {'name':'New','user_id':'another'}])
def test_invalid_profile_does_not_change_account(account, body):
    client, db, user, _ = account
    assert client.patch('/api/auth/profile',json=body).status_code in (400,422)
    assert db.get_user_by_id(user['id'])['name'] == 'Before'

def test_settings_require_authentication(account):
    client, _, _, _ = account
    client.headers.pop('Authorization')
    assert client.patch('/api/auth/profile',json={'name':'New'}).status_code == 401
    assert client.post('/api/auth/change-password',json={'current_password':'old-password','new_password':'new-password'}).status_code == 401

def test_password_change_checks_current_password(account):
    client, db, user, _ = account
    response = client.post('/api/auth/change-password',json={'current_password':'wrong','new_password':'new-password'})
    assert response.status_code == 400
    assert verify_password('old-password', db.get_user_by_email(user['email'])['password_hash'])

@pytest.mark.parametrize('password',['short','é'*37])
def test_password_length_validated_without_server_error(account,password):
    client, db, user, _ = account
    assert client.post('/api/auth/change-password',json={'current_password':'old-password','new_password':password}).status_code in (400,422)
    assert verify_password('old-password',db.get_user_by_email(user['email'])['password_hash'])

def test_password_change_revokes_old_session_and_returns_working_token(account):
    client, db, user, other = account
    time.sleep(.005)
    response = client.post('/api/auth/change-password',json={'current_password':'old-password','new_password':'new-password'})
    assert response.status_code == 200
    assert verify_password('new-password',db.get_user_by_email(user['email'])['password_hash'])
    assert verify_password('other-password',db.get_user_by_email(other['email'])['password_hash'])
    assert client.get('/api/auth/me').status_code == 401
    client.headers['Authorization'] = 'Bearer ' + response.json()['token']
    assert client.get('/api/auth/me').status_code == 200
