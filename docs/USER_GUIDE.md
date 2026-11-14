# Справка GenesisAI: фоновые задания

## Как отменить фоновое задание? How do I cancel a background job?

Для отмены фонового задания используйте HTTP-метод DELETE и endpoint `/api/jobs/{id}`. В `{id}` подставьте `job_id` созданного задания. Заголовок `Authorization: Bearer <session_token>` должен содержать session token владельца задания. Отменить можно своё задание со статусом pending или running. В интерфейсе используйте кнопку Cancel job в карточке задания; после обновления страницы статус cancelled сохраняется.

To cancel a background job, send `DELETE /api/jobs/{id}` with the owner's session token in the Authorization header. Replace `{id}` with the returned job_id. Pending and running jobs can be cancelled. In the chat UI, click Cancel job; the cancelled status persists after refreshing.

## Как узнать статус задания? How do I check job status?

`GET /api/jobs/{id}` возвращает статус своего задания: pending, running, succeeded, failed или cancelled. `GET /api/jobs` показывает список последних своих заданий. Запросы требуют session token владельца. Результат обучения содержит model_id и метрики; завершённая модель доступна в Registry.

Use `GET /api/jobs/{id}` for one owned job or `GET /api/jobs` to list recent jobs. Both require the owner's session token. A completed training result includes model_id and evaluation metrics.

Эта справка сверена с обработчиками `app/routers/jobs.py`.
