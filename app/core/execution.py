"""Ограничение одновременных работ и сериализация диалогов без растущего кэша."""
import contextvars
import functools
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException

EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="genesis")
_capacity = threading.BoundedSemaphore(8)
_submission_slots = threading.BoundedSemaphore(8)
_conversations = [threading.RLock() for _ in range(256)]
_users = [threading.BoundedSemaphore(2) for _ in range(256)]


def _bucket(value):
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 256


def bounded_call(function):
    @functools.wraps(function)
    def call(*args, **kwargs):
        import inspect
        bound = inspect.signature(function).bind(*args, **kwargs)
        user = str(bound.arguments.get("user_id", "internal"))
        if not _capacity.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="The server is busy; retry later", headers={"Retry-After": "5"})
        semaphore = _users[_bucket(user)]
        acquired = semaphore.acquire(blocking=False)
        if not acquired:
            _capacity.release()
            raise HTTPException(status_code=429, detail="Too many simultaneous operations", headers={"Retry-After": "5"})
        try:
            return function(*args, **kwargs)
        finally:
            semaphore.release()
            _capacity.release()
    return call


def serial_conversation(function):
    @functools.wraps(function)
    def call(req, user_id):
        with _conversations[_bucket(user_id + ":" + str(req.conversation_id or "default"))]:
            return function(req, user_id)
    return call


def submit(function, *args):
    context = contextvars.copy_context()
    if not _submission_slots.acquire(blocking=False):
        raise HTTPException(status_code=429,detail="Execution queue is full",headers={"Retry-After":"5"})
    try:
        future = EXECUTOR.submit(context.run, function, *args)
    except BaseException:
        _submission_slots.release()
        raise
    future.add_done_callback(lambda _: _submission_slots.release())
    return future
