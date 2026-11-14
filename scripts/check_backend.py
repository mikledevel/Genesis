"""Запуск тестов в отдельной копии без локальной .env и пользовательских файлов."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="genesis-check-") as work:
        target = Path(work)
        for name in ("app", "tests", "scripts", "static", "docker", "docs", "eval", ".github"):
            shutil.copytree(source / name, target / name, ignore=shutil.ignore_patterns("__pycache__", "node_modules"))
        for name in ("requirements.txt", "requirements.lock.txt", "requirements.render.txt", "pytest.ini", "render.yaml", "requirements.in", "requirements-dev.in"):
            shutil.copy2(source / name, target / name)
        for name in ("datasets", "agents", "blocks", "registry", "marketplace", "generated", "models"):
            (target / name).mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not any(part in k.upper() for part in
               ("API_KEY", "DATABASE_URL", "STRIPE", "SMTP", "SENTRY", "EMAIL_FROM", "RESEND", "PUBLIC_BASE", "RENDER", "GENESIS_TEST"))}
        env.update(DEBUG="true", SECRET_KEY="isolated-check-secret-at-least-32-bytes", DATABASE_URL="",
                   GROQ_API_KEY="gsk_test_placeholder", OPENROUTER_API_KEY="", GEMINI_API_KEY="",
                   STRIPE_SUBSCRIPTIONS_ENABLED="true", CRYPTO_PAYMENTS_ENABLED="false",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        # Только явно переданная локальная тестовая БД может попасть в дочерний процесс.
        if os.environ.get("GENESIS_TEST_POSTGRES_URL"):
            env["GENESIS_TEST_POSTGRES_URL"] = os.environ["GENESIS_TEST_POSTGRES_URL"]
        if os.environ.get("GENESIS_TEST_DOCKER") == "1":
            env["GENESIS_TEST_DOCKER"] = "1"
        result = subprocess.run([sys.executable, "-m", "pytest", *(sys.argv[1:] or ["tests/", "-q", "--tb=short"])], cwd=target, env=env)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
