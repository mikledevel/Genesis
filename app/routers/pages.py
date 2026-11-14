"""pages routes (split out of app/main.py)."""
from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

# ── Pages ──
@router.get("/")
def root(): return FileResponse("static/index.html")

@router.get("/admin")
def admin_page():
    """The page itself is public static HTML with no data; every /api/admin/* call checks for a verified owner e-mail."""
    return FileResponse("static/admin.html")

@router.get("/builder")
def builder(): return FileResponse("static/builder/index.html")


@router.get("/health")
def health(): return {"status": "healthy", "version": "1.0.0"}


@router.get("/ready")
def ready():
    from app.db.database import GenesisDB
    from fastapi import HTTPException
    try:
        with GenesisDB()._connect() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception as error:
        raise HTTPException(status_code=503,detail="Persistent storage is unavailable") from error
    return {"status":"ready"}
