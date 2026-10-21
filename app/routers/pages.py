"""pages routes (split out of app/main.py)."""
from fastapi import APIRouter
from app.deps import (
    FileResponse
)

router = APIRouter()

# ── Pages ──
@router.get("/")
async def root(): return FileResponse("static/index.html")

@router.get("/admin")
async def admin_page():
    """The page itself is public static HTML with no data; every /api/admin/* call checks for a verified owner e-mail."""
    return FileResponse("static/admin.html")

@router.get("/builder")
async def builder(): return FileResponse("static/builder/index.html")


@router.get("/health")
async def health(): return {"status": "healthy", "version": "1.0.0"}
