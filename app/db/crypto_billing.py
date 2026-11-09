"""Durable invoices and access grants for direct USDT TRC20 payments."""
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS crypto_invoice_counters (
    scope TEXT PRIMARY KEY, next_sequence INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS crypto_invoices (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
    plan_id TEXT NOT NULL, recipient TEXT NOT NULL, amount_units INTEGER NOT NULL,
    created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
    paid_at INTEGER, txid TEXT UNIQUE, block_number INTEGER,
    last_check_at INTEGER NOT NULL DEFAULT 0, submitted_txid TEXT,
    UNIQUE(recipient, amount_units)
);
CREATE INDEX IF NOT EXISTS idx_crypto_invoice_user ON crypto_invoices(user_id, created_at);
"""
PERIOD_SECONDS = 30 * 86400


class CryptoBillingMixin:
    def _crypto_user_lock(self, conn, user_id):
        conn.execute('BEGIN' if self.is_postgres else 'BEGIN IMMEDIATE')
        if self.is_postgres:
            conn.execute('SELECT id FROM users WHERE id=? FOR UPDATE', (user_id,))

    def get_crypto_access(self, user_id):
        with self._connect() as conn:
            row = conn.execute('SELECT crypto_plan_id,crypto_access_until FROM users WHERE id=?', (user_id,)).fetchone()
        until = int(row[1] or 0) if row else 0
        return {'plan_id': row[0] if row else None, 'access_until': until,
                'active': bool(row and row[0] in ('pro','business') and until > int(time.time()))}

    @staticmethod
    def _crypto_invoice_row(row):
        if not row:
            return None
        names = ('id','user_id','plan_id','recipient','amount_units','created_at','expires_at',
                 'paid_at','txid','block_number','last_check_at','submitted_txid')
        data = dict(zip(names, row))
        data['status'] = 'paid' if data['paid_at'] else ('expired' if data['expires_at'] < int(time.time()) else 'pending')
        data['amount'] = f"{data['amount_units'] // 1_000_000}.{data['amount_units'] % 1_000_000:06d}"
        data['network'] = 'TRON (TRC20)'
        data['currency'] = 'USDT'
        data['period_days'] = 30
        return data

    def get_crypto_invoice(self, invoice_id, user_id):
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM crypto_invoices WHERE id=? AND user_id=?', (invoice_id,user_id)).fetchone()
        return self._crypto_invoice_row(row)

    def list_crypto_invoices(self, user_id):
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM crypto_invoices WHERE user_id=? ORDER BY created_at DESC,id DESC LIMIT 10', (user_id,)).fetchall()
        return [self._crypto_invoice_row(row) for row in rows]

    def create_crypto_invoice(self, user_id, plan_id, price_usd, recipient):
        now = int(time.time())
        if plan_id not in ('pro','business') or price_usd <= 0:
            raise ValueError('Choose a paid plan')
        with self._connect() as conn:
            self._crypto_user_lock(conn, user_id)
            row = conn.execute('SELECT plan,stripe_subscription_status,crypto_plan_id,crypto_access_until FROM users WHERE id=?', (user_id,)).fetchone()
            if not row:
                raise ValueError('Account was not found')
            if row[0] in ('pro','business') or row[1] in ('active','trialing','past_due','incomplete'):
                raise ValueError('This account already has an externally managed paid plan. Contact support before changing payment methods.')
            if row[2] and (row[3] or 0) > now and row[2] != plan_id:
                raise ValueError('You can renew your current plan now, or choose another plan after its access period ends.')
            existing = conn.execute('SELECT id FROM crypto_invoices WHERE user_id=? AND plan_id=? AND recipient=? AND paid_at IS NULL AND expires_at>? ORDER BY created_at DESC LIMIT 1',
                                    (user_id,plan_id,recipient,now)).fetchone()
            if existing:
                invoice_id = existing[0]
            else:
                pending = conn.execute('SELECT plan_id FROM crypto_invoices WHERE user_id=? AND paid_at IS NULL AND expires_at>? LIMIT 1', (user_id,now)).fetchone()
                if pending:
                    raise ValueError('You already have an unpaid invoice. Reopen it in payment history, or wait for it to expire before selecting another plan.')
                count = conn.execute('SELECT COUNT(*) FROM crypto_invoices WHERE user_id=? AND created_at>=?', (user_id,now-86400)).fetchone()[0]
                if count >= 5:
                    raise ValueError('You can create up to five payment invoices per day. Reopen an existing invoice or try tomorrow.')
                # A never-reused fractional discount ties a transfer to exactly one account.
                # No sender secrets or wallet signatures are required, including for exchange withdrawals.
                serial = conn.execute('INSERT INTO crypto_invoice_counters(scope,next_sequence) VALUES (?,1) ON CONFLICT(scope) DO UPDATE SET next_sequence=crypto_invoice_counters.next_sequence+1 RETURNING next_sequence',
                                      (f'{recipient}:{price_usd}',)).fetchone()[0]
                if serial > 9999:
                    raise ValueError('New crypto invoices are temporarily unavailable. Contact support.')
                invoice_id = 'crypto_' + uuid.uuid4().hex
                conn.execute('INSERT INTO crypto_invoices(id,user_id,plan_id,recipient,amount_units,created_at,expires_at) VALUES (?,?,?,?,?,?,?)',
                             (invoice_id,user_id,plan_id,recipient,price_usd*1_000_000-serial,now,now+3600))
            conn.commit()
        return self.get_crypto_invoice(invoice_id,user_id)

    def begin_crypto_check(self, invoice_id, user_id, txid):
        now = int(time.time())
        with self._connect() as conn:
            cur = conn.execute('UPDATE crypto_invoices SET last_check_at=?,submitted_txid=? WHERE id=? AND user_id=? AND paid_at IS NULL AND last_check_at<=?',
                               (now,txid,invoice_id,user_id,now-10))
        return cur.rowcount == 1

    def apply_crypto_payment(self, invoice_id, user_id, proof):
        now = int(time.time())
        with self._connect() as conn:
            self._crypto_user_lock(conn,user_id)
            invoice = self._crypto_invoice_row(conn.execute('SELECT * FROM crypto_invoices WHERE id=? AND user_id=?', (invoice_id,user_id)).fetchone())
            if not invoice:
                raise ValueError('Invoice was not found')
            if invoice['paid_at']:
                return {'status':'paid', 'already_processed':True}
            if (proof['recipient'] != invoice['recipient'] or proof['amount_units'] != invoice['amount_units']
                    or not invoice['created_at']*1000 <= proof['timestamp_ms'] <= invoice['expires_at']*1000):
                raise ValueError('The confirmed transfer does not match this invoice')
            user = conn.execute('SELECT plan,stripe_subscription_status,crypto_plan_id,crypto_access_until FROM users WHERE id=?', (user_id,)).fetchone()
            if user[0] in ('pro','business') or user[1] in ('active','trialing','past_due','incomplete'):
                raise ValueError('Your billing method changed after this invoice was created. Contact support with the transaction ID.')
            if user[2] and (user[3] or 0) > now and user[2] != invoice['plan_id']:
                raise ValueError('Another plan is already active. Contact support with the transaction ID.')
            # Unique txid plus the locked user/invoice makes retries and races idempotent.
            conn.execute('UPDATE crypto_invoices SET paid_at=?,txid=?,block_number=? WHERE id=?',
                         (now,proof['txid'],proof['block_number'],invoice_id))
            until = max(now,int(user[3] or 0)) + PERIOD_SECONDS
            conn.execute('UPDATE users SET crypto_plan_id=?,crypto_access_until=? WHERE id=?', (invoice['plan_id'],until,user_id))
            conn.commit()
        return {'status':'paid','plan_id':invoice['plan_id'],'access_until':until,'already_processed':False}
