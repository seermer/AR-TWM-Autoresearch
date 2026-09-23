"""The only route from a container to an LLM (spec 4.2, 13.1, 13.3)."""
from __future__ import annotations

import asyncio
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..telemetry.recorder import TelemetryError
from ..tools.context import bearer
from .mock import MockBook
from .store import CallStore

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}
# The OpenAI SDK convention: the base URL includes the API version (spec 2: OPENAI_BASE_URL).
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class Upstream:
    def __init__(self, base_url: str | None, api_key: str, *, timeout_s: float, retries: int,
                 transport: httpx.AsyncBaseTransport | None = None, sleep=asyncio.sleep) -> None:
        # Used as given: "/responses" is appended, so the URL must already end in its version
        # ("/v1"). Appending "/v1" here would break providers whose version path differs.
        self._base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout_s
        self._retries = retries
        self._transport = transport
        self._sleep = sleep

    async def post(self, path: str, body: dict) -> tuple[int, dict, int]:
        # path is "/responses" or "/chat/completions"; the base URL already ends in /v1
        attempts, status, payload = 0, 599, {"error": "no attempt made"}
        async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
            for attempt in range(self._retries + 1):
                attempts = attempt + 1
                try:
                    r = await client.post(self._base + path, json=body, headers=self._headers)
                    status = r.status_code
                    try:
                        payload = r.json() if r.content else {}
                    except ValueError:        # e.g. an HTML error page from a proxy in front of the API
                        payload = {"error": {"message": f"upstream returned HTTP {status} with a "
                                                        f"non-JSON body", "body": r.text[:2000]}}
                        status = status if status >= 400 else 502   # never pass garbage on as success
                except httpx.HTTPError as exc:
                    status, payload = 599, {"error": f"{type(exc).__name__}: {exc}"}
                if status not in RETRYABLE and status != 599:
                    break
                if attempt < self._retries:
                    await self._sleep(2 ** attempt)
        return status, payload, attempts


def create_gateway_app(*, registry, store: CallStore, allowed_models: set[str],
                       upstream: Upstream | None, mocks: MockBook) -> FastAPI:
    # Bounds CallStore's memory to live containers (Task 4 ruling): once a token is
    # revoked, its linking state is dropped along with it.
    registry.on_revoke(store.forget)

    app = FastAPI(title="ar-gateway")

    async def handle(request: Request, endpoint: str, upstream_path: str) -> JSONResponse:
        caller = registry.lookup(bearer(request.headers.get("authorization")))
        if caller is None:
            return JSONResponse({"error": {"message": "unknown or revoked token"}}, status_code=401)
        body = await request.json()
        if body.get("stream"):
            return JSONResponse({"error": {"message": "streaming is not supported by the gateway; "
                                                      "use non-streaming calls"}},
                                status_code=400)
        if body.get("model") not in allowed_models:
            return JSONResponse({"error": {"message": f"model {body.get('model')!r} is not in the "
                                                      f"allowlist {sorted(allowed_models)}"}},
                                status_code=403)
        try:
            meta = store.begin(caller, endpoint, body)
        except TelemetryError as exc:
            return JSONResponse({"error": {"message": f"telemetry unavailable: {exc}"}}, status_code=500)
        started = time.monotonic()
        try:
            if caller.mock_script or upstream is None:
                status, payload, attempts = 200, MockBook.response(
                    body.get("model", ""), mocks.next(caller.mock_script or "smoke", caller.token)), 1
            else:
                status, payload, attempts = await upstream.post(upstream_path, body)
        except Exception as exc:  # noqa: BLE001 -- the request is recorded; its failure must be too
            status, payload, attempts = 502, {"error": {"message": f"gateway: {type(exc).__name__}: {exc}"}}, 1
        try:
            store.end(meta, caller, status=status, body=payload,
                      latency_s=time.monotonic() - started, attempts=attempts)
        except TelemetryError as exc:
            return JSONResponse({"error": {"message": f"telemetry unavailable: {exc}"}}, status_code=500)
        return JSONResponse(payload, status_code=status)

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await handle(request, "/v1/responses", "/responses")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await handle(request, "/v1/chat/completions", "/chat/completions")

    return app
