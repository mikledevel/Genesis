"""Экспорт небольшого однопользовательского staging; в production выключен."""
import gzip
from fastapi import APIRouter, Depends, HTTPException, Response
from app.config import settings
from app.core.auth import get_current_user_id
from app.db.database import GenesisDB

router = APIRouter()


@router.get('/api/pilot/backup')
def pilot_backup(user_id: str = Depends(get_current_user_id)):
    if not settings.genesis_pilot_enabled or not settings.pilot_email:
        raise HTTPException(status_code=404, detail='Not found')
    db = GenesisDB()
    user = db.get_user_by_id(user_id)
    if not user or user['email'].strip().lower() != settings.pilot_email.strip().lower() or not db.is_email_verified(user_id):
        raise HTTPException(status_code=403, detail='Pilot owner only')
    if not db.is_postgres:
        raise HTTPException(status_code=503, detail='PostgreSQL is required')
    from app.core.pilot_snapshot import snapshot_bytes
    try:
        payload = snapshot_bytes(db)
    except ValueError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    return Response(gzip.compress(payload), media_type='application/gzip', headers={
        'Cache-Control': 'no-store',
        'Content-Disposition': 'attachment; filename="genesis-staging-snapshot.json.gz"',
    })
