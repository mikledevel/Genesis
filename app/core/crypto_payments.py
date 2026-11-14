"""Read-only verification of native USDT transfers on TRON mainnet.

No private keys, transaction signing, custody, or transaction broadcasts occur here.
Only solidified receipts from the mainnet Solidity API can grant paid access.
"""
import hashlib
import re
import time
import threading
import httpx
from app.config import settings

TRONGRID = 'https://api.trongrid.io'
USDT_CONTRACT = 'TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t'
TRANSFER_TOPIC = 'ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
HEX64 = re.compile(r'^[0-9a-f]{64}$')
_HEAD_LOCK = threading.Lock()
_HEAD_CACHE = (0.0, 0)


def tron_address_hex(address):
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    try:
        number = 0
        for character in address:
            number = number * 58 + alphabet.index(character)
        raw = number.to_bytes((number.bit_length()+7)//8, 'big')
        raw = b'\0'*(len(address)-len(address.lstrip('1'))) + raw
        if len(raw) != 25 or raw[0] != 0x41:
            raise ValueError()
        checksum = hashlib.sha256(hashlib.sha256(raw[:-4]).digest()).digest()[:4]
        if checksum != raw[-4:]:
            raise ValueError()
        return raw[1:21].hex()
    except (ValueError, TypeError, AttributeError):
        raise ValueError('Invalid TRON receiving address') from None


def crypto_ready():
    if not settings.crypto_payments_enabled:
        return False
    if not settings.debug and not (settings.database_url and settings.public_base_url.startswith('https://')):
        return False
    try:
        tron_address_hex(settings.crypto_usdt_tron_address)
        return True
    except ValueError:
        return False


def _rpc(method, payload):
    headers = {'TRON-PRO-API-KEY': settings.trongrid_api_key} if settings.trongrid_api_key else {}
    try:
        response = httpx.post(f'{TRONGRID}/walletsolidity/{method}', json=payload,
                              headers=headers, timeout=httpx.Timeout(12, connect=5))
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get('Error') or data.get('error'):
            raise RuntimeError()
        return data
    except (httpx.HTTPError, ValueError, RuntimeError):
        raise RuntimeError('Blockchain verification is temporarily unavailable. Your transfer is not lost; retry with the same transaction ID later.') from None


def check_node():
    global _HEAD_CACHE
    with _HEAD_LOCK:
        if time.monotonic() - _HEAD_CACHE[0] < 30:
            return _HEAD_CACHE[1]
        head = _rpc('getnowblock', {})
        raw = head.get('block_header', {}).get('raw_data', {})
        if (type(raw.get('number')) is not int or raw['number'] <= 0
                or type(raw.get('timestamp')) is not int or abs(time.time()*1000-raw['timestamp']) > 600_000):
            raise RuntimeError('The blockchain node is behind. Please create your invoice later.')
        _HEAD_CACHE = (time.monotonic(),raw['number'])
        return raw['number']


def verify_transfer(invoice, txid):
    txid = txid.lower()
    if not HEX64.fullmatch(txid):
        raise ValueError('Enter the 64-character TRON transaction ID')
    receipt = _rpc('gettransactioninfobyid', {'value':txid})
    if not receipt:
        return None  # Submitted/mined is not enough: wait until the receipt is solidified.
    if receipt.get('id', '').lower() != txid:
        raise RuntimeError('The blockchain node returned a different transaction. Retry later.')
    if receipt.get('receipt', {}).get('result') != 'SUCCESS':
        raise ValueError('This transaction did not execute successfully')
    height, timestamp = receipt.get('blockNumber'), receipt.get('blockTimeStamp')
    if type(height) is not int or height <= 0 or type(timestamp) is not int:
        raise RuntimeError('The blockchain receipt is incomplete. Retry later.')
    if not invoice['created_at']*1000 <= timestamp <= invoice['expires_at']*1000:
        raise ValueError('This transfer was made outside this invoice payment window. Contact support if you already sent funds.')
    recipient_hex = tron_address_hex(invoice['recipient'])
    contract_hex = tron_address_hex(USDT_CONTRACT)
    for log in receipt.get('log', []):
        contract = log.get('address', '').lower()
        if contract.startswith('41') and len(contract) == 42:
            contract = contract[2:]
        topics = log.get('topics', [])
        if (contract != contract_hex or len(topics) != 3 or topics[0].lower() != TRANSFER_TOPIC
                or not all(isinstance(t, str) and HEX64.fullmatch(t.lower()) for t in topics)):
            continue
        if topics[2].lower() != '0'*24 + recipient_hex:
            continue
        amount = log.get('data','').lower()
        if not HEX64.fullmatch(amount) or int(amount,16) != invoice['amount_units']:
            continue
        return {'txid':txid,'recipient':invoice['recipient'],'amount_units':int(amount,16),
                'block_number':height,'timestamp_ms':timestamp}
    raise ValueError('No USDT TRC20 transfer matches this invoice address and exact amount. Do not send another payment; check the transaction details first.')

# Discovery is an index lookup only. Every candidate is rechecked against the
# solidified receipt above before any access grant is committed.
_DISCOVERY_LOCK = threading.Lock()
_DISCOVERY_CACHE = {}


def _transfer_page(address, params):
    headers = {'TRON-PRO-API-KEY': settings.trongrid_api_key} if settings.trongrid_api_key else {}
    try:
        response = httpx.get(f'{TRONGRID}/v1/accounts/{address}/transactions/trc20',
                             params=params, headers=headers, timeout=httpx.Timeout(12, connect=5))
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get('success') is not True or not isinstance(data.get('data'), list):
            raise RuntimeError()
        return data
    except (httpx.HTTPError, ValueError, RuntimeError):
        raise RuntimeError('Automatic payment lookup is temporarily unavailable. Keep this invoice and retry later; do not pay again.') from None


def _incoming_transfers(invoice):
    # Invoices last one hour. Two aligned hours cover every invoice in this cohort,
    # so all clients share one cached, bounded lookup rather than hammering the node.
    start = invoice['created_at'] // 3600 * 3600 * 1000
    end = start + 2 * 3600 * 1000
    key = (invoice['recipient'], start, end)
    with _DISCOVERY_LOCK:
        now = time.monotonic()
        cached = _DISCOVERY_CACHE.get(key)
        if cached and cached[0] > now:
            return cached[1]
        params = {'only_confirmed':'true','only_to':'true','contract_address':USDT_CONTRACT,
                  'min_timestamp':start,'max_timestamp':end,'limit':200,'order_by':'block_timestamp,asc'}
        rows, seen = [], set()
        for _ in range(5):
            page = _transfer_page(invoice['recipient'], params)
            rows.extend(page['data'])
            fingerprint = page.get('meta', {}).get('fingerprint')
            if not fingerprint:
                # Cache only complete histories. Never silently treat a truncated
                # or failed pagination response as proof that no payment exists.
                for old_key, value in list(_DISCOVERY_CACHE.items()):
                    if value[0] <= now:
                        del _DISCOVERY_CACHE[old_key]
                if len(_DISCOVERY_CACHE) >= 128:
                    _DISCOVERY_CACHE.pop(next(iter(_DISCOVERY_CACHE)))
                _DISCOVERY_CACHE[key] = (time.monotonic()+20, rows)
                return rows
            if not isinstance(fingerprint,str) or len(fingerprint)>1024 or fingerprint in seen:
                break
            seen.add(fingerprint)
            params['fingerprint'] = fingerprint
        raise RuntimeError('Payment history is too busy for automatic lookup. Retry later or use the optional TXID field.')


def discover_transfer(invoice):
    tron_address_hex(invoice['recipient'])
    for transfer in _incoming_transfers(invoice):
        if not isinstance(transfer,dict):
            continue
        info = transfer.get('token_info') or {}
        txid = str(transfer.get('transaction_id','')).lower()
        value = str(transfer.get('value',''))
        timestamp = transfer.get('block_timestamp')
        if (transfer.get('to') != invoice['recipient'] or transfer.get('type') != 'Transfer'
                or info.get('address') != USDT_CONTRACT or str(info.get('decimals')) != '6'
                or not HEX64.fullmatch(txid) or not value.isdecimal() or len(value)>30
                or int(value) != invoice['amount_units'] or type(timestamp) is not int
                or not invoice['created_at']*1000 <= timestamp <= invoice['expires_at']*1000):
            continue
        try:
            proof = verify_transfer(invoice,txid)
        except ValueError:
            continue  # Index metadata is never sufficient proof of execution.
        if proof:
            return proof
    return None


def check_invoice_automatically(db, invoice):
    if not invoice['paid_at'] and db.begin_crypto_scan(invoice['id'],invoice['user_id']):
        proof = discover_transfer(invoice)
        if proof:
            db.apply_crypto_payment(invoice['id'],invoice['user_id'],proof)
    fresh = db.get_crypto_invoice(invoice['id'],invoice['user_id'])
    return {'status':fresh['status'],'invoice':fresh,'access':db.get_crypto_access(invoice['user_id'])}


def scan_pending_payments(db):
    if not crypto_ready():
        return
    for invoice in db.pending_crypto_invoices():
        try:
            check_invoice_automatically(db,invoice)
        except RuntimeError as error:
            print(f'[crypto] Automatic lookup paused: {error}')
            break  # Provider outage: avoid retrying every invoice in the same tick.
        except (ValueError,) + db._integrity_errors as error:
            print(f"[crypto] Invoice {invoice['id']} requires review: {error}")
