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
from app.routers import auth, pages, chat_agent, marketplace, ml, bots

app = FastAPI(title="Genesis AI", version="1.0.0")
# Order matters: Starlette applies the LAST-added middleware outermost, so it must be added
# last to wrap everything else, including early-exit responses (like a 429 from the rate
# limiter) that never reach the route handler. CORS goes last so those responses still carry
# CORS headers - otherwise a rate-limited response looks like a CORS failure in the browser
# instead of the 429 it actually is.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(GeoBlockMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def _start_background_scheduler():
    # Runs scheduled bot skills (see app/core/scheduler.py) - "Level 3" autonomous checks
    # that fire on a timer rather than in response to a chat message. Safe to start
    # unconditionally: with zero scheduled_jobs rows, the poll loop just wakes up every 30s,
    # finds nothing due, and goes back to sleep.
    from app.core.scheduler import start_scheduler
    start_scheduler()

# Route groups (order = order of first appearance in the old single-file main.py).
app.include_router(auth.router)
app.include_router(pages.router)
app.include_router(chat_agent.router)
app.include_router(marketplace.router)
app.include_router(ml.router)
app.include_router(bots.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
