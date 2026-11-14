"""Шифрование серверных интеграционных токенов; ключ остаётся в окружении."""
import base64
import hashlib
from cryptography.fernet import Fernet


def _cipher():
    from app.config import settings
    material = settings.integration_encryption_key or settings.secret_key
    key = base64.urlsafe_b64encode(hashlib.sha256(("genesis/integrations/v1:" + material).encode()).digest())
    return Fernet(key)


def encrypt_secret(value):
    return "enc:v1:" + _cipher().encrypt(value.encode()).decode()


def decrypt_secret(value):
    return _cipher().decrypt(value[7:].encode()).decode() if value.startswith("enc:v1:") else value


def key_digest(value):
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()
