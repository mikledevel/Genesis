"""Start the production server using Render's assigned public URL and port."""
import os
import sys


if __name__ == "__main__":
    public_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not public_url:
        raise SystemExit("RENDER_EXTERNAL_URL must be set to the service's HTTPS URL")
    if not os.environ.get("ALLOWED_ORIGINS"):
        os.environ["ALLOWED_ORIGINS"] = public_url
    if not os.environ.get("PUBLIC_BASE_URL"):
        os.environ["PUBLIC_BASE_URL"] = public_url
    os.environ["DEBUG"] = "false"
    os.execv(sys.executable, [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "0.0.0.0", "--port", os.environ.get("PORT", "10000"),
        "--workers", "1",
    ])
