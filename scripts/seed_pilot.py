"""Создать единственного участника в выделенном staging без отправки писем."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def seed():
    if os.environ.get('GENESIS_PILOT_ENABLED') != 'true':
        raise RuntimeError('Pilot bootstrap is disabled')
    email = os.environ['PILOT_EMAIL'].strip().lower()
    password = os.environ['PILOT_PASSWORD']
    if len(password) < 24:
        raise RuntimeError('Pilot password must contain at least 24 characters')
    from app.db.database import GenesisDB
    from app.core.auth import hash_password
    db = GenesisDB()
    existing = db.get_user_by_email(email)
    if existing:
        print('Pilot account already exists; credentials unchanged')
        return existing['id']
    user = db.create_user(email, hash_password(password), os.environ.get('PILOT_NAME', 'Mikle'))
    # Staging-учётная запись: доступ выдан владельцем проекта, почтовой рассылки нет.
    db.mark_email_verified(user['id'])
    db.set_user_plan(user['id'], 'pro')
    print('Staging pilot account created')
    return user['id']


if __name__ == '__main__':
    seed()
