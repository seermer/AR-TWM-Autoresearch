"""The only route from a container to an LLM."""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Mapping
from typing import NamedTuple

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import telemetry
from ..budget import Budget
from ..faults import KernelFault
from ..isolation import censor_request
from ..tools.context import bearer
from .mock import MockBook
from .store import CallStore

MAX_SLEEP_S = 60                    # the longest single backoff sleep
PROBE_TTL_S = 60                    # a probe's verdict is reused this long
PROBE_PROMPT = "Reply with OK."
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


def role_label(header: str | None) -> str | None:
    """The X-AR-Role header, safe as a file name."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", (header or "").strip())[:64] or None


def upstream_error(payload) -> dict | None:
    """The error in a body sent with HTTP 200: a router's error object, at the top or inside the first choice,
    or a first choice whose message has neither content nor a tool call (unless it ran out of output:
    asking again would run out again)."""
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    error = next((e for e in (payload.get("error"), first.get("error")) if isinstance(e, dict)), None)
    message = first.get("message")
    if (error is None and isinstance(message, dict) and first.get("finish_reason") != "length"
            and not (message.get("content") or message.get("tool_calls"))):
        return {"code": 502, "message": "the model returned an empty reply"}
    return error


def waited_out(status: int) -> bool:
    """A status the provider recovers from by itself: a timeout, a rate limit, a server or connection error."""
    return status in (408, 429) or status >= 500


class Reply(NamedTuple):
    status: int
    payload: dict
    attempts: int
    gave_up: bool               # the provider kept failing for the whole wait: not the agent's doing


class Upstream:
    """Forwards a request. A timeout, rate limit, server or connection error is retried with backoff for up
    to `wait_s`. Any other 4xx is the agent's when a tiny probe request with the same model succeeds;
    otherwise the provider (or the kernel's own settings) is failing, and it is waited out the same way."""

    def __init__(self, base_url: str | None, api_key: str, *, timeout_s: float, wait_s: float,
                 effort: str | None = None, transport: httpx.AsyncBaseTransport | None = None,
                 sleep=asyncio.sleep) -> None:
        # Used as given; the endpoint path ("/chat/completions", "/responses") is appended.
        # Providers differ: OpenAI's base includes "/v1", others have no version segment.
        self._base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.effort = effort          # reasoning effort forced on every forwarded request
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout_s
        self._wait_s = float(wait_s)
        self._transport = transport
        self._sleep = sleep
        self._probes: dict[tuple, tuple[float, bool]] = {}     # (path, model) -> (when, the provider answered)

    @classmethod
    def from_env(cls, environ: Mapping[str, str], **kwargs) -> "Upstream":
        """OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_EFFORT (empty -> not enforced); `kwargs`
        go to the constructor (timeout_s, wait_s, ...)."""
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

    async def _send(self, client: httpx.AsyncClient, path: str, body: dict) -> tuple[int, dict]:
        with telemetry.span("llm.try", {"url": self._base + path}) as sp:
            try:
                r = await client.post(self._base + path, json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                telemetry.error(exc, "HTTP 599")
                sp.status, sp.output = "error", {"status": 599}
                return 599, {"error": f"{type(exc).__name__}: {exc}"}
            status = r.status_code
            try:
                payload = r.json() if r.content else {}
            except ValueError as exc:     # e.g. an HTML error page from a proxy in front of the API
                telemetry.error(exc, "an error body holding the text")
                payload = {"error": {"message": f"upstream returned HTTP {status} with a "
                                                f"non-JSON body", "body": r.text}}
                status = status if status >= 400 else 502   # never pass garbage on as success
            error = upstream_error(payload)
            if status < 400 and error:                      # a router's 200 around the provider's error
                code = error.get("code")
                status = code if isinstance(code, int) and 400 <= code <= 599 else 502
            sp.status = "ok" if status < 400 else "error"
            sp.output = {"http_status": r.status_code, "status": status, "headers": dict(r.headers), "body": payload}
            return status, payload

    async def _provider_answers(self, client: httpx.AsyncClient, path: str, model) -> bool:
        """Whether a tiny request with the same model and the enforced fields succeeds (cached)."""
        key, now = (path, model), time.monotonic()
        with telemetry.span("llm.probe", {"model": model}) as sp:
            if key in self._probes and now - self._probes[key][0] < PROBE_TTL_S:
                sp.output = {"answers": self._probes[key][1], "cached": True}
                return self._probes[key][1]
            body = {"model": model, **({"messages": [{"role": "user", "content": PROBE_PROMPT}]}
                                       if path == "/chat/completions" else {"input": PROBE_PROMPT})}
            status, payload = await self._send(client, path, self.enforce(path, body))
            self._probes[key] = (now, status < 400)
            sp.output = {"answers": status < 400, "cached": False}
            return status < 400

    async def post(self, path: str, body: dict) -> Reply:
        started, slept, delay, attempts = time.monotonic(), 0.0, 1.0, 0
        async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
            while True:
                attempts += 1
                status, payload = await self._send(client, path, body)
                if status < 400:
                    chosen, why = "return", "a success"
                elif not waited_out(status) and await self._provider_answers(client, path, body.get("model")):
                    chosen, why = "return", "a probe with the same model succeeds: the request is the agent's"
                elif max(slept, time.monotonic() - started) + delay > self._wait_s:
                    chosen, why = "give up", f"the provider kept failing for the whole wait ({self._wait_s:.0f} s)"
                else:
                    chosen, why = "wait", ("a status the provider recovers from" if waited_out(status) else
                                           "the probe failed too: the provider or the kernel's settings fail")
                telemetry.decision("llm.upstream", {"status": status, "tries": attempts, "slept_s": slept},
                                   ["return", "wait", "give up"], chosen, why)
                if chosen != "wait":
                    return Reply(status, payload, attempts, chosen == "give up")
                with telemetry.span("llm.wait", {"delay_s": delay}) as sp:
                    await self._sleep(delay)
                    sp.output = {"slept_s": delay}
                slept, delay = slept + delay, min(delay * 2, MAX_SLEEP_S)


def changes(before: dict, after: dict) -> dict:
    """What a step changed at a request's top level: the keys it set (new or changed) and the keys it removed."""
    return {"set": {k: v for k, v in after.items() if k not in before or before[k] != v},
            "removed": [k for k in before if k not in after]}


def apply_changes(body: dict, changed: dict) -> dict:
    """`body` after a step that made `changed` (see `changes`), keys in the same order."""
    out = {k: v for k, v in body.items() if k not in changed["removed"]}
    out.update(changed["set"])
    return out


def _check(sp, name: str, refused: bool, status: int, message: str | None, **inputs) -> tuple[int, dict] | None:
    """One check of a request, as a decision; a refusal ends `sp` refused with the response."""
    telemetry.decision("llm.check", {"check": name, **inputs}, ["pass", "refuse"], "refuse" if refused else "pass",
                       message if refused else "passed")
    if not refused:
        return None
    payload = {"error": {"message": message}}
    sp.status, sp.output = "refused", {"status": status, "body": payload}
    return status, payload


def create_gateway_app(*, registry, store: CallStore, allowed_models: set[str],
                       upstream: Upstream | None, mocks: MockBook,
                       budget: Budget | None = None, censor: list[str] = (),
                       touch: Callable[[], None] = lambda: None) -> FastAPI:
    """`touch`: called as each call starts and ends."""
    # Bounds CallStore's memory to live containers: once a token is
    # revoked, its linking state is dropped along with it.
    registry.on_revoke(store.forget)

    app = FastAPI(title="ar-gateway")

    async def handle(request: Request, endpoint: str, upstream_path: str) -> JSONResponse:
        caller = registry.lookup(bearer(request.headers.get("authorization")))
        raw = await request.body()
        role = request.headers.get("x-ar-role")
        with telemetry.span("llm.request", {"endpoint": endpoint, "role": role,
                                            "body": raw.decode("utf-8", errors="replace")},
                            parent=caller.span if caller is not None else None) as sp:
            status, payload = await answer(sp, caller, raw, role, endpoint, upstream_path)
        return JSONResponse(payload, status_code=status)

    async def answer(sp, caller, raw: bytes, role, endpoint: str, upstream_path: str) -> tuple[int, dict]:
        """Rejected requests are never forwarded and leave no llm_calls row."""
        if r := _check(sp, "token", caller is None, 401, "unknown or revoked token"):
            return r
        try:
            body = json.loads(raw)
        except ValueError as exc:             # malformed JSON or undecodable bytes
            telemetry.error(exc, "refused: not a JSON object")
            body = None
        if r := _check(sp, "json object", not isinstance(body, dict), 400, "the request body must be a JSON object"):
            return r
        if r := _check(sp, "stream", bool(body.get("stream")), 400,
                       "streaming is not supported by the gateway; use non-streaming calls"):
            return r
        # Images reach the model only through the ask tool, which caps their number and size.
        for name, is_part, message in (("video", _is_video_part, NO_VIDEO_MESSAGE),
                                       ("image", _is_image_part, NO_IMAGE_MESSAGE)):
            if r := _check(sp, name, carries(body, is_part), 400, message):
                return r
        if r := _check(sp, "model", body.get("model") not in allowed_models, 403,
                       f"model {body.get('model')!r} is not in the allowlist {sorted(allowed_models)}",
                       model=body.get("model"), allowed=sorted(allowed_models)):
            return r
        mock = bool(caller.mock_script) or upstream is None
        if not mock and budget is not None:
            why = budget.exhausted()
            if r := _check(sp, "budget", why is not None, 402, why, usd=budget.usd, max_usd=budget.max_usd):
                return r
        if not mock:
            # each step records what it changed: the request's raw text and these rebuild what is forwarded
            with telemetry.span("llm.enforce", {"effort": upstream.effort}) as step:
                before, body = body, upstream.enforce(upstream_path, body)
                step.output = changes(before, body)
        with telemetry.span("llm.censor") as step:
            before, body = body, censor_request(body, list(censor))    # the model and the record get the same
            step.output = changes(before, body)
        meta = store.begin(caller, endpoint, body, role=role_label(role))
        touch()
        gave_up = False
        try:
            if mock:
                render = MockBook.chat_response if upstream_path == "/chat/completions" else MockBook.response
                status, payload = 200, render(
                    body.get("model", ""), mocks.next(caller.mock_script or "smoke", caller.token))
            else:
                status, payload, _, gave_up = await upstream.post(upstream_path, body)
        except Exception as exc:  # noqa: BLE001 -- the request is recorded; its failure must be too
            telemetry.error(exc, "HTTP 502 to the agent")
            status, payload = 502, {"error": {"message": f"gateway: {type(exc).__name__}: {exc}"}}
        cost = budget.record(status, payload.get("usage") if isinstance(payload, dict) else None) \
            if (budget is not None and not mock) else None
        if gave_up:
            caller.fail(f"the LLM provider kept failing (last: HTTP {status}) for the whole wait")
        try:
            store.end(meta, caller, status=status, body=payload, cost_usd=cost, mock=mock)
        except KernelFault as exc:            # the audit reads only what was recorded: the reply is withheld
            telemetry.error(exc, "the reply is withheld (HTTP 500) and the phase stops")
            caller.fail(f"gateway: {exc}")
            payload, status = {"error": {"message": "the kernel could not record this call"}}, 500
        touch()
        sp.status = "ok" if status < 400 else "error"
        sp.output = {"status": status, "body": payload, "cost_usd": cost, "mock": mock}
        return status, payload

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await handle(request, "/v1/responses", "/responses")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await handle(request, "/v1/chat/completions", "/chat/completions")

    return app
