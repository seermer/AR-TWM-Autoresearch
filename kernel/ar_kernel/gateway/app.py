"""The only route from a container to an LLM."""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..budget import Budget
from ..isolation import censor_request
from ..telemetry.recorder import TelemetryError
from ..tools.context import bearer
from .mock import MockBook
from .store import CallStore

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}
DEFAULT_BASE_URL = "https://api.openai.com/v1"
# Chat content parts and Responses input items that carry video. "file"/"input_file" is only
# a video part when its mime type or a data: URL inside it says so (an uploaded file_id has
# no visible mime, so it is let through).
VIDEO_PART_TYPES = {"video_url", "video", "input_video"}
FILE_PART_TYPES = {"file", "input_file"}
IMAGE_PART_TYPES = {"image_url", "image", "input_image"}
NO_VIDEO_MESSAGE = ("video content is not accepted by the gateway; caption clips with the caption_videos "
                    "kernel tool (a GPU job) instead of sending video directly")
NO_IMAGE_MESSAGE = ("image content is not accepted by the gateway; ask about images with the ask kernel "
                    "tool instead of sending them directly")


def _is_video_part(part) -> bool:
    if not isinstance(part, dict):
        return False
    kind = part.get("type")
    if kind in VIDEO_PART_TYPES:
        return True
    return kind in FILE_PART_TYPES and "video/" in json.dumps(part)


def _is_image_part(part) -> bool:
    if not isinstance(part, dict):
        return False
    kind = part.get("type")
    return kind in IMAGE_PART_TYPES or (kind in FILE_PART_TYPES and "image/" in json.dumps(part))


def _has(parts, is_part) -> bool:
    if isinstance(parts, dict):
        parts = [parts]
    return isinstance(parts, list) and any(is_part(p) for p in parts)


def carries(body: dict, is_part) -> bool:
    """True if a chat `messages` content part, or a Responses `input` item (directly or
    inside its `content`), is such a part."""
    for message in body.get("messages") or []:
        if isinstance(message, dict) and _has(message.get("content"), is_part):
            return True
    input_ = body.get("input")
    if isinstance(input_, list):
        for item in input_:
            if isinstance(item, dict) and (is_part(item) or _has(item.get("content"), is_part)):
                return True
    return False


def upstream_error(payload) -> dict | None:
    """The error object a router put in a body it sent with HTTP 200: at the top, or inside the first choice."""
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    return next((e for e in (payload.get("error"), first.get("error")) if isinstance(e, dict)), None)


class Upstream:
    def __init__(self, base_url: str | None, api_key: str, *, timeout_s: float, retries: int,
                 effort: str | None = None, transport: httpx.AsyncBaseTransport | None = None,
                 sleep=asyncio.sleep) -> None:
        # Used as given; the endpoint path ("/chat/completions", "/responses") is appended.
        # Providers differ: OpenAI's base includes "/v1", others have no version segment.
        self._base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.effort = effort          # reasoning effort forced on every forwarded request
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout_s
        self._retries = retries
        self._transport = transport
        self._sleep = sleep

    @classmethod
    def from_env(cls, environ: Mapping[str, str], **kwargs) -> "Upstream":
        """OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_EFFORT (empty -> not enforced); `kwargs`
        go to the constructor (timeout_s, retries, ...)."""
        return cls(environ.get("OPENAI_BASE_URL"), environ.get("OPENAI_API_KEY", ""),
                   effort=environ.get("OPENAI_EFFORT") or None, **kwargs)

    def enforce(self, path: str, body: dict) -> dict:
        """`body` with the configured reasoning effort, overriding whatever the agent sent."""
        if not self.effort:
            return body
        if path == "/chat/completions":
            return {**body, "reasoning_effort": self.effort}
        reasoning = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}
        return {**body, "reasoning": {**reasoning, "effort": self.effort}}

    async def post(self, path: str, body: dict) -> tuple[int, dict, int]:
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
                        # No truncation of telemetry: the full body is recorded,
                        # even though it never reaches the agent JSON-encoded verbatim either way.
                        payload = {"error": {"message": f"upstream returned HTTP {status} with a "
                                                        f"non-JSON body", "body": r.text}}
                        status = status if status >= 400 else 502   # never pass garbage on as success
                    error = upstream_error(payload)
                    if status < 400 and error:                      # a router's 200 around the provider's error
                        code = error.get("code")
                        status = code if isinstance(code, int) and 400 <= code <= 599 else 502
                except httpx.HTTPError as exc:
                    status, payload = 599, {"error": f"{type(exc).__name__}: {exc}"}
                if status not in RETRYABLE and status != 599:
                    break
                if attempt < self._retries:
                    await self._sleep(2 ** attempt)
        return status, payload, attempts


def create_gateway_app(*, registry, store: CallStore, allowed_models: set[str],
                       upstream: Upstream | None, mocks: MockBook,
                       budget: Budget | None = None, censor: list[str] = ()) -> FastAPI:
    # Bounds CallStore's memory to live containers: once a token is
    # revoked, its linking state is dropped along with it.
    registry.on_revoke(store.forget)

    app = FastAPI(title="ar-gateway")

    async def handle(request: Request, endpoint: str, upstream_path: str) -> JSONResponse:
        caller = registry.lookup(bearer(request.headers.get("authorization")))
        if caller is None:
            return JSONResponse({"error": {"message": "unknown or revoked token"}}, status_code=401)
        try:
            body = await request.json()
        except ValueError:                    # malformed JSON or undecodable bytes
            body = None
        if not isinstance(body, dict):        # rejected like the checks below: not recorded, never forwarded
            return JSONResponse({"error": {"message": "the request body must be a JSON object"}}, status_code=400)
        if body.get("stream"):
            return JSONResponse({"error": {"message": "streaming is not supported by the gateway; "
                                                      "use non-streaming calls"}},
                                status_code=400)
        # Rejected like the checks above: not recorded, never forwarded. Images reach the model only
        # through the ask tool, which caps their number and size.
        for is_part, message in ((_is_video_part, NO_VIDEO_MESSAGE), (_is_image_part, NO_IMAGE_MESSAGE)):
            if carries(body, is_part):
                return JSONResponse({"error": {"message": message}}, status_code=400)
        if body.get("model") not in allowed_models:
            return JSONResponse({"error": {"message": f"model {body.get('model')!r} is not in the "
                                                      f"allowlist {sorted(allowed_models)}"}},
                                status_code=403)
        mock = bool(caller.mock_script) or upstream is None
        if not mock and budget is not None:
            why = budget.exhausted()
            if why:       # rejected like the checks above: not recorded, never forwarded
                return JSONResponse({"error": {"message": why}}, status_code=402)
        if not mock:
            body = upstream.enforce(upstream_path, body)   # before begin: record what is forwarded
        body = censor_request(body, list(censor))    # before begin: the model and the record get the same
        try:
            meta = store.begin(caller, endpoint, body)
        except TelemetryError as exc:
            return JSONResponse({"error": {"message": f"telemetry unavailable: {exc}"}}, status_code=500)
        started = time.monotonic()
        try:
            if mock:
                render = MockBook.chat_response if upstream_path == "/chat/completions" else MockBook.response
                status, payload, attempts = 200, render(
                    body.get("model", ""), mocks.next(caller.mock_script or "smoke", caller.token)), 1
            else:
                status, payload, attempts = await upstream.post(upstream_path, body)
        except Exception as exc:  # noqa: BLE001 -- the request is recorded; its failure must be too
            status, payload, attempts = 502, {"error": {"message": f"gateway: {type(exc).__name__}: {exc}"}}, 1
        cost = budget.record(status, payload.get("usage") if isinstance(payload, dict) else None) \
            if (budget is not None and not mock) else None
        try:
            store.end(meta, caller, status=status, body=payload,
                      latency_s=time.monotonic() - started, attempts=attempts,
                      cost_usd=cost, mock=mock)
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
