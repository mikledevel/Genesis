# GenesisAI

Платформа обучения ML-моделей, создания разговорных агентов и продажи доступа через API. Backend — FastAPI, PostgreSQL или локальный SQLite; интерфейс — `static/`. Python 3.12.

## Локальный запуск

```sh
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

В `.env` задайте `DEBUG=true`, собственный стабильный `SECRET_KEY` и `GROQ_API_KEY`. Остальные интеграции подключаются отдельно. Пустой `DATABASE_URL` выбирает SQLite. Не используйте ключи тестового окружения в production.

```sh
python scripts/migrate.py
uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

`/health` проверяет процесс, `/ready` — подключение к БД. OpenAPI доступна в локальном debug-режиме. Настройка deployment: [docs/RENDER_DEPLOYMENT.md](docs/RENDER_DEPLOYMENT.md).

## Проверки

Тестовый runner работает в отдельной копии без `.env` и пользовательских данных:

```sh
python scripts/check_backend.py tests/ -q
cd tests/ui
npm ci
node ui_test.js
node admin_test.js
node antibot_test.js
```

Для интеграционных тестов PostgreSQL передайте `GENESIS_TEST_POSTGRES_URL` с адресом одноразовой БД на loopback. Тесты удаляют содержимое её схемы `genesis`. Для контейнерных проверок:

```sh
docker build -t genesis-sandbox:latest -f docker/sandbox.Dockerfile .
GENESIS_TEST_DOCKER=1 python scripts/check_backend.py tests/test_docker_model_isolation.py -q
```

## Устройство backend

- `app/routers/`, `app/api/contracts.py`: HTTP-контракты, валидация и права доступа.
- `app/core/`: агенты, обучение, платёжные операции, интеграции, ограниченное выполнение.
- `app/db/`: аккаунты, транзакции, документы конфигураций, очередь и сохранённые файлы.
- `app/job_runner.py`: дочерний процесс тяжёлых заданий; `POST /api/jobs`, получение статуса и отмена по ID.
- `scripts/migrate.py`, `scripts/backup_db.py`: миграция и резервное копирование.

Итоги исправлений и эксплуатационные ограничения: [docs/BACKEND_HARDENING.md](docs/BACKEND_HARDENING.md). Первоначальный аудит сохранён в [docs/BACKEND_REVIEW_2026-11-10.md](docs/BACKEND_REVIEW_2026-11-10.md).

Контракты инструментов, источники ответов, трассировка, фоновые задачи и воспроизводимые evals: [docs/LLM_IMPROVEMENTS.md](docs/LLM_IMPROVEMENTS.md). Для новой таблицы traces нужна миграция схемы до версии 4.

Подготовка staging и пилота: [docs/PILOT_READINESS_2026-11-14.md](docs/PILOT_READINESS_2026-11-14.md), [сценарий пилота](docs/PILOT.md).
