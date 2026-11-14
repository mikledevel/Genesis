"""Automatic invoice recovery while the application is awake.

Render Free may sleep when idle. Unpaid invoices remain in PostgreSQL and are
rescanned when the service wakes or a customer reopens the payment window.
"""
import asyncio

_task = None


async def _monitor_loop():
    from app.db.database import GenesisDB
    from app.core.crypto_payments import scan_pending_payments
    db = GenesisDB()
    while True:
        try:
            await asyncio.to_thread(scan_pending_payments,db)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f'[crypto] Background payment lookup failed: {error}')
        await asyncio.sleep(30)


def start_crypto_monitor():
    from app.config import settings
    global _task
    if not settings.crypto_payments_enabled:
        return None
    if _task is None or _task.done():
        _task = asyncio.create_task(_monitor_loop())
    return _task


def stop_crypto_monitor():
    global _task
    if _task is not None:
        _task.cancel()
        _task = None
