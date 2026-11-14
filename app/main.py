import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.deps import (
    CORSMiddleware,
    FastAPI,
    GeoBlockMiddleware,
    RateLimitMiddleware,
    StaticFiles,
    _allowed_origins
)
from app.routers import auth, pages, chat_agent, marketplace, ml, bots, admin, crypto, jobs

import asyncio
from contextlib import asynccontextmanager
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.observability import RequestContextMiddleware, http_error, unexpected_error, input_error


@asynccontextmanager
async def lifespan(app):
    from app.db.database import GenesisDB
    from app.core.scheduler import start_scheduler, stop_scheduler
    from app.core.crypto_monitor import start_crypto_monitor, stop_crypto_monitor
    from app.core.background import start_workers, stop_workers
    db = await asyncio.to_thread(GenesisDB)
    await asyncio.to_thread(db.recover_usage)
    start_scheduler()
    start_crypto_monitor()
    start_workers()
    try:
        yield
    finally:
        stop_scheduler()
        stop_crypto_monitor()
        await stop_workers()


app = FastAPI(title="Genesis AI", version="1.1.0", lifespan=lifespan)
app.add_exception_handler(StarletteHTTPException, http_error)
app.add_exception_handler(Exception, unexpected_error)
app.add_exception_handler(ValueError, input_error)
app.add_middleware(RequestContextMiddleware)
# Order matters: Starlette applies the LAST-added middleware outermost, so it must be added
# last to wrap everything else, including early-exit responses (like a 429 from the rate
# limiter) that never reach the route handler. CORS goes last so those responses still carry
# CORS headers - otherwise a rate-limited response looks like a CORS failure in the browser
# instead of the 429 it actually is.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(GeoBlockMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route groups (order = order of first appearance in the old single-file main.py).
app.include_router(auth.router)
app.include_router(crypto.router)
app.include_router(pages.router)
app.include_router(chat_agent.router)
app.include_router(marketplace.router)
app.include_router(ml.router)
app.include_router(bots.router)
app.include_router(admin.router)
app.include_router(jobs.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
