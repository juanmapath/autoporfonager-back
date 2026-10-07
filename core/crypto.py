import base64
import hashlib
import json
from typing import Any, Dict
from cryptography.fernet import Fernet, MultiFernet
from django.conf import settings


def _ensure_fernet_key(key: str | bytes) -> bytes:
    """
    Ensures the given key is a valid 32 url-safe base64-encoded byte key.
    If not, deterministically derives a valid 32-byte Fernet key using SHA-256.
    """
    if not key:
        return Fernet.generate_key()

    if isinstance(key, str):
        key_bytes = key.strip().encode("utf-8")
    else:
        key_bytes = key

    try:
        decoded = base64.urlsafe_b64decode(key_bytes)
        if len(decoded) == 32:
            return key_bytes
    except Exception:
        pass

    # Deterministic SHA-256 derivation into 32 urlsafe base64 bytes
    digest = hashlib.sha256(key_bytes).digest()
    return base64.urlsafe_b64encode(digest)


def _get_fernet() -> MultiFernet:
    key_setting = getattr(settings, "MASTER_ENCRYPTION_KEY", None)
    if not key_setting:
        key_setting = "autoportonager-master-encryption-key-secret-fallback"

    if isinstance(key_setting, str):
        keys = [k.strip() for k in key_setting.split(",") if k.strip()]
    else:
        keys = [key_setting]

    fernets = [Fernet(_ensure_fernet_key(k)) for k in keys]
    return MultiFernet(fernets)


def encrypt_json(data: Dict[str, Any]) -> bytes:
    """Encrypts a dictionary payload into Fernet ciphertext bytes."""
    raw_bytes = json.dumps(data).encode("utf-8")
    return _get_fernet().encrypt(raw_bytes)


def decrypt_json(cipher_bytes: bytes) -> Dict[str, Any]:
    """Decrypts Fernet ciphertext bytes back into a dictionary."""
    if not cipher_bytes:
        return {}
    decrypted = _get_fernet().decrypt(bytes(cipher_bytes))
    return json.loads(decrypted.decode("utf-8"))
