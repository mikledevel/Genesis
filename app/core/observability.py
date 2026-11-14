"""Идентификаторы запросов, структурированные события и безопасные ошибки."""
import contextvars
import json
import logging
import time
import uuid
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

request_id = contextvars.ContextVar("request_id", default="")
log = logging.getLogger("genesis.http")


def _validate_json(value, depth=0):
    if depth > 16:
        raise ValueError("JSON nesting limit exceeded")
    if isinstance(value, str) and len(value) > 100000:
        raise ValueError("A string exceeds the input limit")
    if isinstance(value, list) and len(value) > 10000:
        raise ValueError("An array exceeds the input limit")
    if isinstance(value, (list, dict)):
        for item in (value.values() if isinstance(value, dict) else value):
            _validate_json(item, depth+1)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = uuid.uuid4().hex
        token = request_id.set(rid)
        started = time.monotonic()
        try:
            if request.headers.get("content-type", "").split(";")[0] == "application/json":
                data = bytearray()
                async for chunk in request.stream():
                    data.extend(chunk)
                    if len(data) > 1_000_000:
                        return JSONResponse({"detail": "JSON body exceeds 1 MB", "code": "input_too_large", "request_id": rid}, status_code=413)
                request._body = bytes(data)
                try:
                    _validate_json(json.loads(data))
                except (ValueError, UnicodeError):
                    return JSONResponse({"detail": "Invalid or excessive JSON input", "code": "invalid_input", "request_id": rid}, status_code=422)
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            log.info(json.dumps({"request_id": rid, "method": request.method, "route": getattr(request.scope.get("route"), "path", "unmatched"),
                                 "status": response.status_code, "duration_ms": round((time.monotonic()-started)*1000)}))
            return response
        finally:
            request_id.reset(token)


async def http_error(request, error):
    code = {400: "bad_request", 401: "unauthorized", 402: "insufficient_balance", 403: "forbidden", 404: "not_found", 409: "conflict", 422: "invalid_input", 429: "rate_limited", 503: "unavailable"}.get(error.status_code, "request_failed")
    return JSONResponse({"detail": error.detail, "code": code, "request_id": request_id.get()}, status_code=error.status_code, headers=error.headers)


async def unexpected_error(request, error):
    import traceback
    frames = [{"file":f.filename,"line":f.lineno,"function":f.name} for f in traceback.extract_tb(error.__traceback__)]
    log.error(json.dumps({"event":"request_failed","request_id":request_id.get(),"type":type(error).__name__,"frames":frames}))
    return JSONResponse({"detail": "An internal error occurred", "code": "internal_error", "request_id": request_id.get()}, status_code=500)


async def input_error(request,error):
    return JSONResponse({"detail":"Invalid input","code":"invalid_input","request_id":request_id.get()},status_code=422)
