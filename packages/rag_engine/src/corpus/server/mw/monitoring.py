"""
FastAPI middleware for request monitoring and metrics collection.
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from tools.monitoring.metrics import ERRORS, LATENCY, REQUESTS, monitor_operation


class MonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        method = request.method
        component = "http"

        with monitor_operation(component, f"{method} {path}") as span:
            try:
                response = await call_next(request)
                span.set_attribute("http.status_code", response.status_code)
                return response
            except Exception as e:
                span.set_attribute("error", str(e))
                raise
