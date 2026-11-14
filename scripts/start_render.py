"""Start the production server using Render's assigned public URL and port."""
import os
import sys
import subprocess


if __name__ == "__main__":
    public_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not public_url:
        raise SystemExit("RENDER_EXTERNAL_URL must be set to the service's HTTPS URL")
    if not os.environ.get("ALLOWED_ORIGINS"):
        os.environ["ALLOWED_ORIGINS"] = public_url
    if not os.environ.get("PUBLIC_BASE_URL"):
        os.environ["PUBLIC_BASE_URL"] = public_url
    os.environ["DEBUG"] = "false"
    if os.environ.get("GENESIS_BOOTSTRAP_SCHEMA") == "true":
        # Только для новой выделенной БД, где приложение владеет схемой.
        import psycopg
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            connection.execute("CREATE SCHEMA IF NOT EXISTS genesis")
            connection.execute("REVOKE ALL ON SCHEMA genesis FROM PUBLIC")
    subprocess.run([sys.executable,"scripts/migrate.py"],check=True,
                   env=dict(os.environ,DATABASE_AUTO_MIGRATE="true"))
    os.environ["DATABASE_AUTO_MIGRATE"] = "false"
    if os.environ.get("GENESIS_PILOT_ENABLED") == "true":
        subprocess.run([sys.executable, "scripts/seed_pilot.py"], check=True)
    os.execv(sys.executable, [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "0.0.0.0", "--port", os.environ.get("PORT", "10000"),
        "--workers", "1",
        "--no-proxy-headers",
    ])
