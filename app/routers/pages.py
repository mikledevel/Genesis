"""pages routes (split out of app/main.py)."""
from fastapi import APIRouter
from app.deps import (
    FileResponse
)

router = APIRouter()

# ── Pages ──
@router.get("/")
async def root(): return FileResponse("static/index.html")

@router.get("/builder")
async def builder(): return FileResponse("static/builder/index.html")


@router.get("/health")
async def health(): return {"status": "healthy", "version": "1.0.0"}
